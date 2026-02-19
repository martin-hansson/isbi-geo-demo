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

# --- 1. Global Configuration ---
st.set_page_config(page_title="GEO Search Simulator", layout="wide")

# Configure Ollama & Local Embeddings
Settings.llm = Ollama(model="gemini-3-flash-preview", request_timeout=300.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

if not os.path.exists("./data"):
    os.makedirs("./data")

# --- 2. Helper Functions ---

def get_urls_from_sitemap(sitemap_url):
    """Recursively parses XML sitemaps and sitemap indexes for URLs."""
    try:
        response = requests.get(sitemap_url)
        root = ET.fromstring(response.content)
        
        # Dynamically extract the XML namespace
        match = re.match(r'\{.*\}', root.tag)
        namespace = {'ns': match.group(0)[1:-1]} if match else {}
        ns_prefix = 'ns:' if namespace else ''

        urls = []
        # If it's a WordPress Sitemap Index, recursively fetch the sub-sitemaps
        if 'sitemapindex' in root.tag:
            for loc in root.findall(f'.//{ns_prefix}loc', namespace):
                if loc.text:
                    urls.extend(get_urls_from_sitemap(loc.text))
        else:
            # If it's a standard urlset, grab the actual page URLs
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
                    # Extract URL from the hidden comment on the first line for citations
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
        "from the retrieved 'Context' web pages when relevant.\n\n"
        "RULES:\n"
        "1. Answer naturally and directly. Do NOT start your response with robotic phrases like 'According to the provided context' or 'Based on the context'.\n"
        "2. If the user asks a conversational question ('Hello') or general knowledge, answer using your internal knowledge fluidly.\n"
        "3. If the user asks about specific topics found in the context, seamlessly weave the facts into your answer.\n"
        "4. CITATIONS: When you use information from the context, you MUST include the source URL as an inline citation at the end of the sentence or claim, like this: [https://example.com/...]. Use the 'source' field from the context metadata for the URL."
    )
    
    st.session_state.chat_engine = st.session_state.index.as_chat_engine(
        chat_mode="condense_plus_context",
        system_prompt=system_prompt,
        similarity_top_k=3,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
        verbose=True
    )

async def run_crawl_and_index(urls):
    """Crawls URLs, saves them as .md files to overwrite old data, and returns updated Documents."""
    async with AsyncWebCrawler(verbose=False) as crawler:
        results = await crawler.arun_many(urls)
        for result in results:
            if result.success:
                # Create a safe, deterministic filename from the URL
                safe_name = result.url.replace("https://", "").replace("http://", "").replace("/", "_").strip("_")
                if not safe_name:
                    safe_name = "index"
                filepath = os.path.join("./data", f"{safe_name}.md")
                
                # Write to file, injecting the URL as a hidden HTML comment on the first line
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"<!-- URL: {result.url} -->\n")
                    f.write(result.markdown)
    
    # After crawling and saving, load EVERYTHING currently in the data folder
    return load_data_folder()

# --- Startup Initialization ---
if "engine_initialized" not in st.session_state:
    initial_docs = load_data_folder()
    if initial_docs:
        initialize_engine(initial_docs)
    st.session_state.engine_initialized = True

# --- 3. Sidebar UI ---

with st.sidebar:
    st.header("🌐 Search Engine Indexer")
    st.write("Simulate a crawler indexing target websites.")
    sitemap_url = st.text_input("Enter Sitemap URL", placeholder="https://example.com/wp-sitemap.xml")
    
    if st.button("🚀 Crawl & Rebuild Index"):
        if sitemap_url:
            with st.status("Indexing Web Data...") as status:
                st.write("Extracting URLs (recursively checking indexes)...")
                urls = get_urls_from_sitemap(sitemap_url)
                
                # Filter out obvious non-HTML assets if any slipped through
                urls = [u for u in urls if not u.endswith(('.jpg', '.png', '.pdf'))]
                
                if urls:
                    st.write(f"Crawling {len(urls)} actual web pages...")
                    # Crawl will overwrite matched files and return the full updated folder
                    docs = asyncio.run(run_crawl_and_index(urls))
                    
                    st.write("Generating Vector Embeddings...")
                    initialize_engine(docs)
                    
                    status.update(label="Search Engine Ready!", state="complete")
                    st.success(f"Indexed {len(docs)} total pages successfully.")
                else:
                    st.warning("No valid URLs found.")
        else:
            st.error("Please provide a sitemap URL.")

# --- 4. Chat Interface ---

st.title("🔎 Simulated AI Search")
st.markdown("Ask general questions, or test if the engine retrieves your crawled sites!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Search the web or ask a question..."):
    # Show user input immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Handle the Assistant's streaming response
    with st.chat_message("assistant"):
        if "chat_engine" in st.session_state:
            with st.spinner("Thinking..."):
                # Using stream_chat() instead of chat() to get a generator. 
                # This blocks until the first token is generated.
                streaming_response = st.session_state.chat_engine.stream_chat(prompt)
            
            # We use st.empty() to create a placeholder that we update block by block
            response_placeholder = st.empty()
            full_response = ""
            
            # Iterate through the chunks as Ollama generates them
            for chunk in streaming_response.response_gen:
                full_response += chunk
                # Adding a "▌" gives a nice typing cursor effect
                response_placeholder.markdown(full_response + "▌")
            
            # Finalize the text without the cursor
            response_placeholder.markdown(full_response)
            
            # GEO Feature: Show what the search engine retrieved
            if streaming_response.source_nodes:
                with st.expander("⚙️ What the engine retrieved (GEO Analysis)"):
                    st.write("These are the pages the vector database thought were most relevant. *If the AI didn't use them, your site needs better optimization!*")
                    unique_sources = set()
                    for node in streaming_response.source_nodes:
                        src = node.node.metadata.get("source")
                        score = node.score if node.score else 0.0
                        if src:
                            unique_sources.add((src, score))
                    
                    # Sort by relevance score
                    for src, score in sorted(unique_sources, key=lambda x: x[1], reverse=True):
                        st.write(f"- [{src}]({src}) *(Relevance Score: {score:.2f})*")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            # Fallback if no index is loaded yet, act as a standard LLM
            with st.spinner("Thinking..."):
                # Stream the fallback response too
                fallback_response_stream = Settings.llm.stream_complete(prompt)
                
            response_placeholder = st.empty()
            full_response = ""
            
            for chunk in fallback_response_stream:
                full_response += chunk.delta
                response_placeholder.markdown(full_response + "▌")
                
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.info("💡 Note: No sitemap has been indexed yet. I am answering using only my base knowledge.")