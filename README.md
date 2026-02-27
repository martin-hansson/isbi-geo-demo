# FakeGPT: A RAG-Powered Chatbot Demo

FakeGPT is a Streamlit-based web application that demonstrates a simple implementation of a Retrieval-Augmented Generation (RAG) chatbot. It allows you to input an XML sitemap URL, crawls the website's content, and builds a local vector index. The chatbot then uses this index to answer user queries with domain-specific facts, falling back to general knowledge when necessary.

It utilizes **LlamaIndex** for RAG orchestration, **Crawl4AI** for asynchronous web scraping, **HuggingFace** for embeddings, and **Ollama** for the LLM.

> [!NOTE]
> This demo doesn't use the internal logic for the chat engine, thus it doesn't rewrite queries for better context. We simulate a RAG engine for clarity with three "agents". This allows us to demonstrate each decision in the pipeline for transparency. The agents are:
>
> 1. Intent agent that decides if the user query and last conversation history requires additional sources.
> 2. A RAG agent that retrieves context based on the intent classification and cosine similarity.
> 3. A response agent that uses the conversation history, eventual retrieved context and query to return a response.

## Prerequisites

- Python 3.8 or higher.
- An accessible Ollama instance. _Note: If you have a local Ollama instance, don't forget to_ `Ollama pull <model_name>`.

## Setup Instructions

Follow these steps to set up the project locally.

1. **Create a Virtual Environment**

It is highly recommended to use a virtual environment to manage dependencies. Open your terminal in the project directory and run:

```
python -m venv isbi-demo
```

2. **Activate the Virtual Environment**

```
source isbi-demo/bin/activate
```

3. **Install Dependencies**

With the virtual environment activated, install the required Python packages using the provided requirements.txt file:

```
pip install -r requirements.txt
```

4. **Run the Application**

Once the packages are installed, you can start the Streamlit application by running:

```
streamlit run fakegpt.py
```

This will start a local server, and your default web browser should automatically open to the app (usually at http://localhost:8501).

### Usage

1. **Rebuild Index:** In the sidebar, enter the URL of a valid XML sitemap (e.g., `https://example.com/sitemap.xml`) and click **Rebuild Index**. The app will crawl the pages, generate vector embeddings, and save the data locally to a `./storage` directory.

2. **Chat:** Use the main chat interface to ask questions. If the question requires specific knowledge from your indexed website, the chatbot will retrieve the relevant text chunks, synthesize an answer, and provide source links.
