import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

st.set_page_config(page_title="FakeGPT", layout="wide")
st.title("What can I help with?")

Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

with st.sidebar:
    st.header("Data Management")
    if st.button("Re-index Crawled Sites"):
        with st.spinner("Indexing markdown files..."):
            documents = SimpleDirectoryReader("data").load_data()
            st.session_state.index = VectorStoreIndex.from_documents(documents)
            st.success("Index ready!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything"):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    if "index" in st.session_state:
        query_engine = st.session_state.index.as_query_engine(streaming=True)
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            response = query_engine.query(prompt)
            for chunk in response.response_gen:
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })
    else:
        st.error("Please index the data in the sidebar first!")