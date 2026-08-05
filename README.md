# Generative Engine Optimization (GEO) Lab

Welcome to the GEO Lab! In this lab, you will explore how Large Language Models (LLMs), Agentic Search Loops, and Retrieval-Augmented Generation (RAG) combine to simulate how modern AI search engines work.

## Prerequisites

Before starting the lab, you must install the following dependencies on your machine:

### 1. Install Node.js
You must have Node.js (version 18 or higher) installed.
- **Mac/Windows:** Download and install from the [official Node.js website](https://nodejs.org/).
- **Linux:** We recommend using nvm (Node Version Manager) or your package manager:
  ```bash
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
  nvm install 20
  ```

### 2. Install Ollama (Local AI Engine)
This lab uses local LLMs to process data securely and for free.
- **Mac:** Download the Mac installer from [Ollama's website](https://ollama.com/).
- **Windows:** Download the Windows preview installer from Ollama.
- **Linux:** Run the automated install script:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

### 3. Download the Required AI Models
Once Ollama is installed and running, open your terminal (or Command Prompt) and pull the specific models required for this lab:

```bash
# 1. Download the generative chatbot model (Llama 3.2 3B)
ollama pull llama3.2:3b

# 2. Download the vector embedding model (Gemma)
ollama pull embeddinggemma
```

*(Note: Depending on your internet connection, downloading the models may take a few minutes as they are several gigabytes in size).*

---

## Running the Lab

Once all prerequisites are installed, you can start the lab application. 

1. **Open your terminal** and navigate to this project folder.
2. **Install project dependencies:**
   ```bash
   npm install
   ```
3. **Start the development server:**
   ```bash
   npm run dev
   ```
4. **Open your browser** and navigate to the URL provided in the terminal (usually `http://localhost:5173`).

---

## Troubleshooting

- **Server Crash / ECONNREFUSED:** Ensure Ollama is running in the background. If the Node.js server crashes, simply restart it by pressing `Ctrl+C` and running `npm run dev` again.
- **Crawler 403 Errors:** If you see 403 Forbidden errors in the terminal, it means the website's firewall (like Cloudflare) blocked the automated web crawler. This is normal when scraping the live web. The LLM will fall back to other successful pages.
- **Empty Local Index:** If your local cosine similarity scores are extremely low, ensure you have generated your `local_index.json` using the correct `embeddinggemma` model.
