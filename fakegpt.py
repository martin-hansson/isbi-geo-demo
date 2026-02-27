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

# You can set the model you want to use. If Ollama is run locally, leave base_url blank.
Settings.llm = Ollama(model="gemini-3-flash-preview", base_url="https://ollama-haproxy.dsv.su.se/", request_timeout=300.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

if not os.path.exists("./data"):
    os.makedirs("./data")


# TODO: Make sure that it can handle recursive crawling without a sitemap.
def get_urls_from_sitemap(sitemap_url):
    """Recursively parses XML sitemaps and sitemap indexes for URLs."""
    try:
        response = requests.get(sitemap_url)

        # Build element tree from XML sitemap content, handling namespaces if present.
        root = ET.fromstring(response.content)
        match = re.match(r'\{.*\}', root.tag)
        namespace = {'ns': match.group(0)[1:-1]} if match else {}
        ns_prefix = 'ns:' if namespace else ''

        # Populate urls list from the sitemap.
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
                    
                    # Appends a document node with source URL metadata for citations.
                    docs.append(Document(
                        text=content,
                        metadata={"source": url, "title": filename}
                    ))
    return docs


def initialize_engine(docs):
    """Builds the vector index, saves it to disk, and initializes the chat engine."""

    # Parses markdown documents into chunks and builds a vector index, then saves it to disk for future sessions.
    # This is a simpler parser, so it could include images and other non-text content in the chunks.
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(docs)
    st.session_state.index = VectorStoreIndex(nodes)
    
    st.session_state.index.storage_context.persist(persist_dir="./storage")


def load_engine_from_storage():
    """Loads a pre-computed index from disk and initializes the chat engine."""
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    st.session_state.index = load_index_from_storage(storage_context)


def intent_agent(query: str):
    """Intent agent: decides whether external indexed sources are needed."""
    
    # Append the last 4 converstation turns to help intent agent decide if sources are needed.
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

    # Instruction prompt for intent classification. The LLM must decide if the query requires sources.
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

    # Change similarity_top_k to adjust how many sources are retrieved.
    # Change similarity_cutoff to adjust minimum cosine similarity threshold.
    retriever = st.session_state.index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(query)
    processor = SimilarityPostprocessor(similarity_cutoff=0.7)
    nodes = processor.postprocess_nodes(nodes)

    if not nodes:
        return {
            "use_sources": False,
            "reason": "no_retrieval_hits",
            "best_score": 0.0,
            "second_score": 0.0,
            "source_nodes": [],
        }

    # Sorts retrieved nodes by their cosine similarity.
    scored_nodes = sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)
    best_score = scored_nodes[0].score or 0.0
    second_score = scored_nodes[1].score or 0.0 if len(scored_nodes) > 1 else 0.0

    return {
        "use_sources": True,
        "reason": "relevance_confident",
        "best_score": best_score,
        "second_score": second_score,
        "source_nodes": scored_nodes,
    }


def response_agent(prompt: str, rag_result: dict):
    """Response agent: generates the final user answer, with or without source context."""
    response_placeholder = st.empty()
    full_response = ""

    # Adds conversation history to the prompt for better context.
    history_messages = st.session_state.get("messages", [])
    recent_history = history_messages[-8:]
    conversation_lines = []
    for msg in recent_history:
        role = msg.get("role", "user").capitalize()
        content = (msg.get("content") or "").strip()
        if content:
            conversation_lines.append(f"{role}: {content}")
    conversation_context = "\n".join(conversation_lines)

    # If we use sources, we want to include the retrieved chunks in the instruction prompt.
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

        # Instruction prompt with retrieved context, conversation history and user query.
        blended_prompt = (
            "You are a highly capable, helpful, and honest AI assistant. Your goal is to provide clear, accurate, and comprehensive answers.\n\n"
            "CORE DIRECTIVES:\n"
            "1. Synthesis: Answer fluidly and conversationally. NEVER expose your internal mechanics. Do not say things like 'Based on the provided snippets', 'According to the context', or 'The sources say'. Synthesize the information naturally as if it were your own knowledge.\n"
            "2. Citations: When stating a specific fact, figure, or claim derived from the provided website context, strictly use an inline citation immediately after the claim (e.g., [https://example.com/page]).\n"
            "3. Formatting: Structure your response for maximum readability. Use **bold text** for emphasis, bullet points for lists, and brief paragraphs.\n"
            "4. Knowledge Blending: If the provided context does not contain the complete answer, seamlessly blend it with your general knowledge, but clearly distinguish hard facts from general advice.\n\n"
            "CONVERSATION HISTORY:\n"
            f"{conversation_context if conversation_context else '(no prior turns)'}\n\n"
            "WEBSITE CONTEXT (Supporting Evidence):\n"
            f"{chr(10).join(context_lines)}\n\n"
            f"USER QUERY: {prompt}\n\n"
            "ASSISTANT RESPONSE:"
        )

        # Stream the response from the LLM, updating the answer in real-time as chunks arrive.
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

    # If we do not use sources, we generate the response with only the conversation history and user query.
    fallback_prompt = (
        "You are a highly capable, helpful, and honest AI assistant. Answer the user's question thoughtfully and accurately based on your general knowledge.\n\n"
        "CORE DIRECTIVES:\n"
        "1. Tone: Maintain a polite, objective, and conversational tone.\n"
        "2. Clarity: Keep answers concise but comprehensive. Break complex ideas into digestible points.\n"
        "3. Formatting: Structure your response for maximum readability. Use **Markdown formatting** extensively, including bolding for key terms, bullet points for lists, and clear paragraph breaks.\n"
        "4. Context: Rely on the conversation history to understand the exact context and intent of the user's latest query.\n\n"
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

        # This is where we include the retrieved sources in the UI.
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