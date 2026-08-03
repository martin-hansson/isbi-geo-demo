"""
FakeGPT: A RAG-Powered Chatbot Demo
This lab demonstrates a simple implementation of a RAG-powered chatbot.
The goal is to adapt your site's content so that the chatbot uses your website as a knowledge source.
"""

import streamlit as st
import asyncio
import requests
import xml.etree.ElementTree as ET
import re
import os
import shutil

from ddgs import DDGS
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from crawl4ai import AsyncWebCrawler

st.set_page_config(page_title="FakeGPT", page_icon="/Users/martin/Developer/isbi-geo-demo/media/gpt-logo.svg", layout="wide")

# You can set the model you want to use. If Ollama is run locally, leave base_url blank.
Settings.llm = Ollama(model="llama3.2:3b", base_url="http://localhost:11434", request_timeout=300.0)
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


def initialize_engine(docs):
    """Builds the vector index, saves it to disk, and initializes the chat engine."""
    # Use SentenceSplitter for predictable chunk sizes (better for GEO)
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(docs)
    st.session_state.index = VectorStoreIndex(nodes)
    st.session_state.index.storage_context.persist(persist_dir="./storage")


def load_engine_from_storage():
    """Loads a pre-computed index from disk."""
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    st.session_state.index = load_index_from_storage(storage_context)


def query_condense_agent(query: str):
    """Condenses the query based on conversation history."""
    history_messages = st.session_state.get("messages", [])
    recent_history = history_messages[-7:-1]

    
    if not recent_history:
        return query
        
    conversation_lines = []
    for msg in recent_history:
        role = msg.get("role", "user").capitalize()
        content = (msg.get("content") or "").strip()
        if content:
            if role == "Assistant" and len(content) > 300:
                content = content[:300] + "..."
            conversation_lines.append(f"{role}: {content}")
    conversation_context = "\n".join(conversation_lines)

    condense_prompt = (
        "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n"
        "If the follow up question is already a standalone question or a general greeting, just return it as is.\n"
        "Do not answer the question, just rewrite it.\n\n"
        "Chat History:\n"
        f"{conversation_context}\n\n"
        f"Follow Up Input: {query}\n"
        "Standalone question:"
    )

    try:
        rewritten_query = Settings.llm.complete(condense_prompt).text.strip()
        return rewritten_query if rewritten_query else query
    except Exception:
        return query


def retrieve_sources(query: str):
    """Retrieves relevant chunks from the index."""
    if "index" not in st.session_state:
        return []
        
    retriever = st.session_state.index.as_retriever(similarity_top_k=10)
    nodes = retriever.retrieve(query)
    return sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)


def search_web_ddg(query: str, max_results=10):
    """Searches the web using DuckDuckGo."""
    try:
        results = DDGS().text(query, max_results=max_results)
        return list(results) if results else []
    except Exception as e:
        return []


def start_new_chat_session():
    """Starts a fresh in-memory chat session."""
    st.session_state.messages = []
    st.session_state.active_chat_session = st.session_state.get("active_chat_session", 0) + 1


async def run_crawl_and_index(urls):
    """Crawls URLs with concurrency limits, saves metadata correctly, and returns Documents."""
    docs = []
    # Limit concurrency so we don't overwhelm the target server
    semaphore = asyncio.Semaphore(5)
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        async def fetch(url):
            async with semaphore:
                return await crawler.arun(url=url)
                
        tasks = [fetch(u) for u in urls]
        # Use return_exceptions=True so one failed page doesn't crash the whole run
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                st.error(f"Failed to crawl {url}: {result}")
                continue
                
            if result and getattr(result, "success", False):
                safe_name = url.replace("https://", "").replace("http://", "").replace("/", "_").strip("_") or "index"
                filepath = os.path.join("./data", f"{safe_name}.md")
                
                # Save purely for inspection purposes
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(result.markdown)
                
                # Create the document with proper metadata natively
                docs.append(Document(
                    text=result.markdown,
                    metadata={"source": url, "title": safe_name}
                ))
                
    return docs


if "engine_initialized" not in st.session_state:
    if os.path.exists("./storage"):
        load_engine_from_storage()
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
                    
                    if docs:
                        st.write("Generating Vector Embeddings...")
                        initialize_engine(docs) 
                        status.update(label="Search Engine Ready!", state="complete")
                        st.success(f"Indexed {len(docs)} total pages successfully.")
                    else:
                        status.update(label="No documents indexed", state="error")
                        st.warning("Crawling failed to return any documents.")
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
        response_placeholder = st.empty()
        full_response = ""
        source_nodes = []
        
        # 1. Query Condensation (Agentic step)
        with st.spinner("Analyzing intent..."):
            rewritten_query = query_condense_agent(prompt)
            
        if "index" in st.session_state:
            # 2. Retrieval
            with st.spinner(f"Searching for: '{rewritten_query}'..."):
                source_nodes = retrieve_sources(rewritten_query)

        # 2.5 Web Search
        web_results = []
        with st.spinner(f"Searching the web for: '{rewritten_query}'..."):
            web_results = search_web_ddg(rewritten_query)

        # 3. Generation (RAG or Fallback)
        history_messages = st.session_state.get("messages", [])
        recent_history = history_messages[-8:-1] # exclude the latest user prompt
        conversation_lines = []
        for msg in recent_history:
            role = msg.get("role", "user").capitalize()
            content = (msg.get("content") or "").strip()
            if content:
                conversation_lines.append(f"{role}: {content}")
        conversation_context = "\n".join(conversation_lines)
        
        merged_sources = []
        if source_nodes or web_results:
            with st.spinner("Merging and ranking sources..."):
                query_emb = Settings.embed_model.get_query_embedding(rewritten_query)
                
                for node in source_nodes:
                    src = node.node.metadata.get("source", "Unknown source")
                    title = node.node.metadata.get("title", "Local Document")
                    score = node.score if node.score else 0.0
                    snippet = re.sub(r"\s+", " ", node.node.get_content()).strip()[:700]
                    merged_sources.append({"type": "local", "src": src, "title": title, "score": score, "snippet": snippet})
                    
                for res in web_results:
                    src = res.get("href", "Unknown web source")
                    title = res.get("title", "Web Page")
                    snippet = res.get("body", "")[:700]
                    try:
                        res_emb = Settings.embed_model.get_text_embedding(snippet)
                        dot = sum(a*b for a, b in zip(query_emb, res_emb))
                        norm1 = sum(a*a for a in query_emb) ** 0.5
                        norm2 = sum(b*b for b in res_emb) ** 0.5
                        score = dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0
                    except Exception:
                        score = 0.0
                    merged_sources.append({"type": "web", "src": src, "title": title, "score": score, "snippet": snippet})
                    
                merged_sources = sorted(merged_sources, key=lambda x: x["score"], reverse=True)[:3]

        with st.spinner("Generating response..."):
            try:
                if merged_sources:
                    # RAG Prompt
                    context_lines = []
                    urls = []
                    urls_set = set()
                    
                    for item in merged_sources:
                        prefix = "Web Source URL" if item["type"] == "web" else "Source URL"
                        context_lines.append(f"Title: {item['title']}\n{prefix}: {item['src']}\nContent: {item['snippet']}\n")
                        if item['src'] and item['src'] not in urls_set:
                            urls.append({"title": item['title'], "url": item['src']})
                            urls_set.add(item['src'])
                    
                    blended_prompt = (
                        "You are a highly capable, helpful, and honest AI assistant. Your goal is to provide clear, accurate, and comprehensive answers.\n\n"
                        "CORE DIRECTIVES:\n"
                        "1. Synthesis: Answer fluidly and conversationally. Synthesize the information naturally as if it were your own knowledge.\n"
                        "2. Citations: You MUST use inline markdown links to cite the sources provided. Format: [Title](URL). Example: 'Berlin has great spots ([Wikipedia](https://wikipedia.org)).'. Do not use numbered citations like [1] or list sources at the end.\n"
                        "3. Relevance: If a source is completely irrelevant to the user's query, ignore it entirely and do not cite it.\n"
                        "4. Formatting: Structure your response for maximum readability. Use **bold text** for emphasis, bullet points for lists, and brief paragraphs.\n"
                        "5. Knowledge Blending: If the provided context does not contain the complete answer, seamlessly blend it with your general knowledge.\n\n"
                        "CONVERSATION HISTORY:\n"
                        f"{conversation_context if conversation_context else '(no prior turns)'}\n\n"
                        "WEBSITE CONTEXT (Supporting Evidence):\n"
                        f"{chr(10).join(context_lines)}\n\n"
                        f"USER QUERY: {prompt}\n\n"
                        "ASSISTANT RESPONSE:"
                    )
                    
                    streaming_response = Settings.llm.stream_complete(blended_prompt)
                    for chunk in streaming_response:
                        full_response += chunk.delta
                        response_placeholder.markdown(full_response + "▌")
                        
                    # Inject sources at the end if the LLM forgot
                    if urls and not re.search(r"\]\(https?://", full_response):
                        citation_line = ", ".join(f"[{item['title']}]({item['url']})" for item in urls[:3])
                        if citation_line:
                            full_response = f"{full_response}\n\nSources: {citation_line}"
                            
                    response_placeholder.markdown(full_response)
                    
                    with st.expander("Retrieved sources"):
                        st.write("These are the text chunks that were the most relevant to the query.")
                        for i, item in enumerate(merged_sources, start=1):
                            type_label = "Local Source" if item["type"] == "local" else "Web Search"
                            st.markdown(f"**{i}. [{item['title']}]({item['src']})** *({type_label}, Cosine similarity: {item['score']:.2f})*")
                            st.info(item['snippet'])
                            st.divider()
                            
                    pipeline_desc = f"Pipeline: Merged local & web sources ➜ Selected top {len(merged_sources)} ➜ Generated RAG response."
                    if rewritten_query != prompt:
                        pipeline_desc = f"Pipeline: Rewrote query to '{rewritten_query}' ➜ Merged sources ➜ Selected top {len(merged_sources)} ➜ Generated RAG response."
                    st.caption(pipeline_desc)
                        
                else:
                    # Fallback Prompt (No context)
                    fallback_prompt = (
                        "You are a highly capable, helpful, and honest AI assistant. Answer the user's question thoughtfully and accurately based on your general knowledge.\n\n"
                        "CONVERSATION HISTORY:\n"
                        f"{conversation_context if conversation_context else '(no prior turns)'}\n\n"
                        f"USER QUERY: {prompt}\n\n"
                        "ASSISTANT RESPONSE:"
                    )
                    
                    streaming_response = Settings.llm.stream_complete(fallback_prompt)
                    for chunk in streaming_response:
                        full_response += chunk.delta
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                    
                    if "index" not in st.session_state:
                        st.caption("Pipeline: No index loaded ➜ Generated base knowledge response.")
                    else:
                        if rewritten_query != prompt:
                            st.caption(f"Pipeline: Rewrote query to '{rewritten_query}' ➜ No relevant sources found ➜ Generated base knowledge response.")
                        else:
                            st.caption("Pipeline: No relevant sources found ➜ Generated base knowledge response.")
                            
            except Exception as e:
                response_placeholder.error(f"Error during generation: {e}")
                full_response = str(e)

        st.session_state.messages.append({"role": "assistant", "content": full_response})