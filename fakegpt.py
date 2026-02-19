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

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from crawl4ai import AsyncWebCrawler

st.set_page_config(page_title="FakeGPT", layout="wide")
Settings.llm = Ollama(model="gemini-3-flash-preview", request_timeout=300.0)
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
    """Builds the vector index and initializes the chat engine."""
    st.session_state.index = VectorStoreIndex.from_documents(docs)
    
    system_prompt = (
        "You are an AI Web Search Assistant. You answer questions naturally, incorporating information "
        "from retrieved web pages only when that context is clearly relevant.\n\n"
        "RULES:\n"
        "1. Answer naturally and directly. Do NOT start with phrases like 'According to the provided context'.\n"
        "2. For conversational or general-knowledge questions, answer directly from internal knowledge.\n"
        "3. Use retrieved context only when it is clearly relevant to the user question.\n"
        "4. If retrieved context is weak/irrelevant, ignore it and answer normally.\n"
        "5. CITATIONS: Only when you actually use context, include inline URL citations like [https://example.com/...]."
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
    router_prompt = (
        "You are a routing classifier for a chatbot with optional web-index tools. "
        "Return exactly one label: NEEDS_SOURCES or NO_SOURCES.\n\n"
        "Choose NEEDS_SOURCES only if the user likely needs facts from specific websites "
        "(business details, pages, offerings, events, policies, location-specific details).\n"
        "Choose NO_SOURCES for greetings, opinions, creative tasks, coding help, math, language tasks, "
        "or broad general-knowledge questions.\n\n"
        f"User query: {query}\n"
        "Label:"
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
    response_placeholder.markdown("_Thinking…_")

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
            "You are a helpful chatbot. Answer naturally and conversationally.\n"
            "Use the optional website snippets as supporting evidence when relevant, but do not over-focus on them.\n"
            "If this is a recommendation query, provide several alternatives naturally (ideally 3-6 when possible), "
            "and do not steer to only one venue unless the user explicitly asks for one."
            "When you use website-derived facts, add inline URL citations like [https://...].\n"
            "If the snippets are only partially relevant, combine with general knowledge and clearly separate uncertain claims.\n\n"
            "Conversation history from the current session:\n"
            f"{conversation_context if conversation_context else '(no prior turns)'}\n\n"
            f"Latest user query:\n{prompt}\n\n"
            "Website snippets (optional evidence):\n"
            + "\n\n".join(context_lines)
        )

        with st.spinner("Thinking..."):
            streaming_response = Settings.llm.stream_complete(blended_prompt)

        for chunk in streaming_response:
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

    with st.spinner("Thinking..."):
        fallback_prompt = (
            "You are a helpful chatbot. Use the conversation history to maintain context within this session.\n"
            "Do not mention internal memory rules. Answer naturally.\n\n"
            "Conversation history from the current session:\n"
            f"{conversation_context if conversation_context else '(no prior turns)'}\n\n"
            f"Latest user query:\n{prompt}\n"
        )
        fallback_response_stream = Settings.llm.stream_complete(fallback_prompt)

    for chunk in fallback_response_stream:
        delta = getattr(chunk, "delta", "")
        full_response += delta
        response_placeholder.markdown(full_response)

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

    st.divider()
    developer_mode = st.toggle("Developer mode", value=False, key="developer_mode")

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

        if developer_mode and agent_result["mode"] == "rag" and agent_result["source_nodes"]:
            with st.expander("Retrieved sources"):
                st.write("These are the pages that were the most relevant to the query.")
                unique_sources = set()
                for node in agent_result["source_nodes"]:
                    src = node.node.metadata.get("source")
                    score = node.score if node.score else 0.0
                    if src:
                        unique_sources.add((src, score))
                for src, score in sorted(unique_sources, key=lambda x: x[1], reverse=True):
                    st.write(f"- [{src}]({src}) *(Relevance Score: {score:.2f})*")

        if developer_mode:
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