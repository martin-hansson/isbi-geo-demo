"""
FakeGPT: A RAG-Powered Chatbot Demo
This lab demonstrates a simple implemtation of a RAG-powered chatbot.
The goal is to adapt your site's content so that the chatbot uses your website is used as a knowledge source.
"""

import streamlit as st
import asyncio
import requests
import xml.etree.ElementTree as ET
import re
import os
import shutil

from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from crawl4ai import AsyncWebCrawler

st.set_page_config(page_title="FakeGPT", layout="wide")
Settings.llm = Ollama(model="gemini-3-flash-preview", base_url="https://ollama-haproxy.dsv.su.se/", request_timeout=300.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

if not os.path.exists("./data"):
    os.makedirs("./data")


def get_urls_from_sitemap(sitemap_url):
    """Recursively parses XML sitemaps and sitemap indexes for URLs."""
    try:
        response = requests.get(sitemap_url)
        root = ET.fromstring(response.content)
        match = re.match(r'\{.*\}', root.tag)
        namespace = {'ns': match.group(0)[1:-1]} if match else {}
        ns_prefix = 'ns:' if namespace else ''

        urls = []
        if 'sitemapindex' in root.tag:
            for loc in root.findall(f'.//{ns_prefix}loc', namespace):
                if loc.text:
                    urls.extend(get_urls_from_sitemap(loc.text))
        else:
            for loc in root.findall(f'.//{ns_prefix}loc', namespace):
                if loc.text:
                    urls.append(loc.text)
        return urls
    except Exception as e:
        st.error(f"Error parsing sitemap {sitemap_url}: {e}")
        return []


def load_data_folder():
    """Reads markdown files from the data folder and reconstructs LlamaIndex Documents."""
    docs = []
    if os.path.exists("./data"):
        for filename in os.listdir("./data"):
            if filename.endswith(".md"):
                filepath = os.path.join("./data", filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    first_line = content.split('\n')[0]
                    url = first_line.replace("<!-- URL: ", "").replace(" -->", "").strip() if first_line.startswith("<!-- URL:") else "Local File"
                    
                    docs.append(Document(
                        text=content,
                        metadata={"source": url, "title": filename}
                    ))
    return docs


def initialize_engine(docs):
    """Builds the vector index, saves it to disk, and initializes the chat engine."""
    
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(docs)
    st.session_state.index = VectorStoreIndex(nodes)
    
    st.session_state.index.storage_context.persist(persist_dir="./storage")
    
    system_prompt = (
        "You are a highly capable AI assistant, designed to be helpful, harmless, and honest. "
        "You provide structured, highly readable answers using Markdown formatting (bolding, lists, etc.).\n\n"
        "CORE DIRECTIVES:\n"
        "1. Synthesis: Answer fluidly. Never expose your internal mechanics by saying 'According to the context provided'.\n"
        "2. Knowledge: Answer general knowledge questions confidently. Use retrieved context only when it specifically addresses the user's prompt.\n"
        "3. Citations: When utilizing factual data from retrieved context, strictly use inline citations (e.g., [https://example.com/page]).\n"
        "4. Clarity: Keep answers concise but comprehensive. Break complex ideas into digestible points."
    )
    
    st.session_state.chat_engine = st.session_state.index.as_chat_engine(
        chat_mode="condense_plus_context",
        system_prompt=system_prompt,
        similarity_top_k=3,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
        verbose=True
    )


def load_engine_from_storage():
    """Loads a pre-computed index from disk and initializes the chat engine."""
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    st.session_state.index = load_index_from_storage(storage_context)
    
    system_prompt = (
        "You are a highly capable AI assistant, designed to be helpful, harmless, and honest. "
        "You provide structured, highly readable answers using Markdown formatting (bolding, lists, etc.).\n\n"
        "CORE DIRECTIVES:\n"
        "1. Synthesis: Answer fluidly. Never expose your internal mechanics by saying 'According to the context provided'.\n"
        "2. Knowledge: Answer general knowledge questions confidently. Use retrieved context only when it specifically addresses the user's prompt.\n"
        "3. Citations: When utilizing factual data from retrieved context, strictly use inline citations (e.g., [https://example.com/page]).\n"
        "4. Clarity: Keep answers concise but comprehensive. Break complex ideas into digestible points."
    )
    
    st.session_state.chat_engine = st.session_state.index.as_chat_engine(
        chat_mode="condense_plus_context",
        system_prompt=system_prompt,
        similarity_top_k=3,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
        verbose=True
    )


def intent_agent(query: str):
    """Intent agent: decides whether external indexed sources are needed."""
    
    history_messages = st.session_state.get("messages", [])
    recent_history = history_messages[-4:] 
    conversation_lines = []
    for msg in recent_history:
        role = msg.get("role", "user").capitalize()
        content = (msg.get("content") or "").strip()
        if content:
            if role == "Assistant" and len(content) > 200:
                content = content[:200] + "..."
            conversation_lines.append(f"{role}: {content}")
    conversation_context = "\n".join(conversation_lines)

    router_prompt = (
        "You are a strict routing classifier for an AI assistant. Your job is to decide if the user's latest query requires searching a specific website index for facts.\n\n"
        "RULES:\n"
        "1. Output ONLY the exact word 'NEEDS_SOURCES' or 'NO_SOURCES'. Do not explain your reasoning.\n"
        "2. Output 'NEEDS_SOURCES' if the user is asking about specific business details, products, services, events, documentation, or anything that requires up-to-date, domain-specific facts.\n"
        "3. Output 'NO_SOURCES' for general knowledge, coding help, math, creative writing, greetings, or casual conversation.\n\n"
        "CONVERSATION CONTEXT:\n"
        f"{conversation_context if conversation_context else '(No prior context)'}\n\n"
        f"LATEST USER QUERY: {query}\n\n"
        "LABEL:"
    )

    try:
        decision = Settings.llm.complete(router_prompt).text.strip().upper()
        if "NEEDS_SOURCES" in decision:
            return {"needs_sources": True, "reason": "intent_llm_requested_sources"}
        return {"needs_sources": False, "reason": "intent_llm_skipped_sources"}
    except Exception:
        return {"needs_sources": False, "reason": "intent_router_error_fallback"}


def rag_agent(
    query: str,
    intent_result: dict
):
    """RAG agent: runs retrieval confidence checks and decides if index should be used."""
    if "index" not in st.session_state:
        return {
            "use_sources": False,
            "reason": "no_index_loaded",
            "best_score": 0.0,
            "second_score": 0.0,
            "source_nodes": [],
        }

    if not intent_result.get("needs_sources", False):
        return {
            "use_sources": False,
            "reason": intent_result.get("reason", "intent_no_sources"),
            "best_score": 0.0,
            "second_score": 0.0,
            "source_nodes": [],
        }

    retriever = st.session_state.index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(query)
    if not nodes:
        return {
            "use_sources": False,
            "reason": "no_retrieval_hits",
            "best_score": 0.0,
            "second_score": 0.0,
            "source_nodes": [],
        }

    scored_nodes = sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)
    selected_nodes = scored_nodes[:3]
    best_score = scored_nodes[0].score or 0.0
    second_score = scored_nodes[1].score or 0.0 if len(scored_nodes) > 1 else 0.0

    return {
        "use_sources": True,
        "reason": "relevance_confident",
        "best_score": best_score,
        "second_score": second_score,
        "source_nodes": selected_nodes,
    }


def response_agent(prompt: str, rag_result: dict):
    """Response agent: generates the final user answer, with or without source context."""
    response_placeholder = st.empty()
    full_response = ""

    history_messages = st.session_state.get("messages", [])
    recent_history = history_messages[-8:]
    conversation_lines = []
    for msg in recent_history:
        role = msg.get("role", "user").capitalize()
        content = (msg.get("content") or "").strip()
        if content:
            conversation_lines.append(f"{role}: {content}")
    conversation_context = "\n".join(conversation_lines)

    if rag_result.get("use_sources", False):
        source_nodes = rag_result.get("source_nodes", [])
        context_lines = []
        urls = []
        for idx, node in enumerate(source_nodes, start=1):
            src = node.node.metadata.get("source", "Unknown source")
            score = node.score if node.score else 0.0
            snippet = re.sub(r"\s+", " ", node.node.get_content()).strip()[:700]
            context_lines.append(f"[{idx}] URL: {src} | score={score:.2f}\n{snippet}")
            if src and src not in urls:
                urls.append(src)

        blended_prompt = (
            "You are a helpful, intelligent, and highly capable AI assistant. Your goal is to provide clear, accurate, and comprehensive answers.\n\n"
            "INSTRUCTIONS:\n"
            "- Answer directly and conversationally. DO NOT say things like 'Based on the provided snippets' or 'The sources say'. Synthesize the information naturally as if it were your own knowledge.\n"
            "- Use formatting extensively to make your answer easy to read. Use **bold text** for emphasis, bullet points for lists, and brief paragraphs.\n"
            "- If the query asks for recommendations or ideas, provide a well-structured list with 3-6 distinct, varied options.\n"
            "- When you state a specific fact, figure, or claim derived from the website snippets, immediately follow it with an inline citation like [https://...].\n"
            "- If the snippets do not contain the complete answer, seamlessly blend them with your general knowledge, but clearly distinguish facts from general advice.\n\n"
            "CONVERSATION HISTORY:\n"
            f"{conversation_context if conversation_context else '(no prior turns)'}\n\n"
            "WEBSITE CONTEXT (Supporting Evidence):\n"
            f"{chr(10).join(context_lines)}\n\n"
            f"USER QUERY: {prompt}\n\n"
            "ASSISTANT RESPONSE:"
        )

        streaming_response = Settings.llm.stream_complete(blended_prompt)
        response_iterator = iter(streaming_response)

        with st.spinner("Thinking..."):
            try:
                first_chunk = next(response_iterator)
            except StopIteration:
                first_chunk = None

        if first_chunk:
            full_response += getattr(first_chunk, "delta", "")
            response_placeholder.markdown(full_response)

        for chunk in response_iterator:
            full_response += getattr(chunk, "delta", "")
            response_placeholder.markdown(full_response)

        if urls and not re.search(r"\[https?://[^\]]+\]", full_response):
            citation_line = " ".join(f"[{url}]" for url in urls[:3])
            if citation_line:
                full_response = f"{full_response}\n\nSources: {citation_line}"

        response_placeholder.markdown(full_response)
        return {
            "content": full_response,
            "mode": "rag",
            "source_nodes": source_nodes,
        }

    fallback_prompt = (
        "You are a helpful, intelligent, and highly capable AI assistant. Answer the user's question thoughtfully and accurately.\n\n"
        "INSTRUCTIONS:\n"
        "- Maintain a polite, objective, and conversational tone.\n"
        "- Structure your response for readability. Use **Markdown formatting**, including bolding for key terms, bullet points for lists, and clear paragraph breaks.\n"
        "- Rely on the conversation history to understand the context of this specific turn.\n\n"
        "CONVERSATION HISTORY:\n"
        f"{conversation_context if conversation_context else '(no prior turns)'}\n\n"
        f"USER QUERY: {prompt}\n\n"
        "ASSISTANT RESPONSE:"
    )
    fallback_response_stream = Settings.llm.stream_complete(fallback_prompt)
    fallback_iterator = iter(fallback_response_stream)

    with st.spinner("Thinking..."):
        try:
            first_chunk = next(fallback_iterator)
        except StopIteration:
            first_chunk = None

    if first_chunk:
        delta = getattr(first_chunk, "delta", "")
        full_response += delta
        response_placeholder.markdown(full_response)

    for chunk in fallback_iterator:
        delta = getattr(chunk, "delta", "")
        full_response += delta
        response_placeholder.markdown(full_response)

    return {
        "content": full_response,
        "mode": "base",
        "source_nodes": [],
    }


def start_new_chat_session():
    """Starts a fresh in-memory chat session (does not persist across browser sessions)."""
    st.session_state.messages = []
    st.session_state.active_chat_session = st.session_state.get("active_chat_session", 0) + 1


async def run_crawl_and_index(urls):
    """Crawls URLs, saves them as .md files to overwrite old data, and returns updated Documents."""
    async with AsyncWebCrawler(verbose=False) as crawler:
        results = await crawler.arun_many(urls)
        for result in results:
            if result.success:
                safe_name = result.url.replace("https://", "").replace("http://", "").replace("/", "_").strip("_")
                if not safe_name:
                    safe_name = "index"
                filepath = os.path.join("./data", f"{safe_name}.md")
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"<!-- URL: {result.url} -->\n")
                    f.write(result.markdown)
    
    return load_data_folder()


if "engine_initialized" not in st.session_state:
    if os.path.exists("./storage"):
        load_engine_from_storage()
    else:
        initial_docs = load_data_folder()
        if initial_docs:
            initialize_engine(initial_docs)
            
    st.session_state.engine_initialized = True

with st.sidebar:
    if st.button("New chat", use_container_width=True):
        start_new_chat_session()
        st.rerun()
    
    st.divider()

    st.header("Search Engine")
    st.write("Input your sitemap URL to include your website in the search index.")
    sitemap_url = st.text_input("Enter Sitemap URL", placeholder="https://example.com/wp-sitemap.xml")
    
    if st.button("Rebuild Index"):
        if sitemap_url:
            with st.status("Indexing Web Data...") as status:
                st.write("Extracting URLs (recursively checking indexes)...")
                urls = get_urls_from_sitemap(sitemap_url)
                urls = [u for u in urls if not u.endswith(('.jpg', '.png', '.pdf'))]
                
                if urls:
                    if os.path.exists("./storage"):
                        st.write("Clearing old index from storage...")
                        shutil.rmtree("./storage")
                        
                    st.write(f"Crawling {len(urls)} actual web pages...")
                    docs = asyncio.run(run_crawl_and_index(urls))
                    
                    st.write("Generating Vector Embeddings...")
                    initialize_engine(docs) 
                    
                    status.update(label="Search Engine Ready!", state="complete")
                    st.success(f"Indexed {len(docs)} total pages successfully.")
                else:
                    st.warning("No valid URLs found.")
        else:
            st.error("Please provide a sitemap URL.")

st.title("What can I help with?")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_chat_session" not in st.session_state:
    st.session_state.active_chat_session = 1

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        intent_result = intent_agent(prompt)
        rag_result = rag_agent(prompt, intent_result)
        agent_result = response_agent(prompt, rag_result)

        if agent_result["mode"] == "rag" and agent_result["source_nodes"]:
            with st.expander("Retrieved sources"):
                st.write("These are the text chunks that were the most relevant to the query.")
                
                for i, node in enumerate(agent_result["source_nodes"], start=1):
                    src = node.node.metadata.get("source", "Unknown source")
                    score = node.score if node.score else 0.0
                    chunk_text = node.node.get_content().strip()
                    
                    st.markdown(f"**{i}. [{src}]({src})** *(Cosine similarity: {score:.2f})*")
                    
                    st.info(chunk_text)
                    st.divider()

        if agent_result["mode"] == "rag":
            st.caption(
                "Pipeline: intent_agent ➜ rag_agent ➜ response_agent | "
                f"used indexed sources ({rag_result['reason']}; best={rag_result['best_score']:.2f}, second={rag_result['second_score']:.2f})."
            )
        else:
            if rag_result["reason"] == "no_index_loaded":
                st.info("💡 Note: No sitemap has been indexed yet. I am answering using only my base knowledge.")
            else:
                st.caption(
                    "Pipeline: intent_agent ➜ rag_agent ➜ response_agent | "
                    f"answered normally ({rag_result['reason']}; best={rag_result['best_score']:.2f}, second={rag_result['second_score']:.2f})."
                )

        st.session_state.messages.append({"role": "assistant", "content": agent_result["content"]})