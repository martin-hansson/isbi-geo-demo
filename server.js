import express from "express";
import cors from "cors";
import * as cheerio from "cheerio";
import { Ollama } from "ollama";
import fs from "fs/promises";
import path from "path";
import { CheerioCrawler } from "crawlee";

const app = express();
app.use(cors());
app.use(express.json({ limit: "50mb" }));

import os from "os";
let FRONTEND_DIR =
  global.FRONTEND_DIR ||
  process.env.FRONTEND_DIR ||
  path.join(process.cwd(), "dist");
let DATA_DIR =
  global.DATA_DIR || process.env.DATA_DIR || path.join(process.cwd(), "data");

// Bulletproof fallback: If it resolved to the root directory /data (which crashes on macOS), override it safely.
if (DATA_DIR === "/data") {
  DATA_DIR = path.join(os.homedir(), ".chatlab_data");
}
const LOCAL_INDEX_FILE = path.join(DATA_DIR, "local_index.json");
const ONLINE_CACHE_FILE = path.join(DATA_DIR, "index.json");
const CHATS_FILE = path.join(DATA_DIR, "chats.json");

// Ensure the DATA_DIR exists (critical when running inside Electron)
import fsSync from "fs";
if (!fsSync.existsSync(DATA_DIR)) {
  fsSync.mkdirSync(DATA_DIR, { recursive: true });
}

const ollama = new Ollama({ host: "http://localhost:11434" });

// Serve the compiled Vite static assets
app.use(express.static(FRONTEND_DIR));

const EMBED_MODEL = "nomic-embed-text";
const SIMILARITY_THRESHOLD = 0.1;

function cosineSimilarity(vecA, vecB) {
  if (!vecA || !vecB) return 0;
  let dotProduct = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Ensure data directory and files exist
async function initDataStorage() {
  try {
    await fs.mkdir(DATA_DIR, { recursive: true });
  } catch (e) {}
  try {
    await fs.access(LOCAL_INDEX_FILE);
  } catch (e) {
    await fs.writeFile(LOCAL_INDEX_FILE, JSON.stringify({}));
  }
  try {
    await fs.access(ONLINE_CACHE_FILE);
  } catch (e) {
    await fs.writeFile(ONLINE_CACHE_FILE, JSON.stringify({}));
  }
  try {
    await fs.access(CHATS_FILE);
  } catch (e) {
    await fs.writeFile(CHATS_FILE, JSON.stringify([]));
  }
}
initDataStorage();

// Read/Write Helpers
async function readJsonFile(filepath) {
  try {
    const data = await fs.readFile(filepath, "utf-8");
    return JSON.parse(data);
  } catch (e) {
    return filepath === CHATS_FILE ? [] : {};
  }
}
async function writeJsonFile(filepath, data) {
  await fs.writeFile(filepath, JSON.stringify(data, null, 2));
}

// --- INDEX SERVICE ---
app.post("/api/crawl", async (req, res) => {
  const { url } = req.body;
  if (!url) return res.status(400).json({ error: "URL is required" });

  try {
    console.log(`Starting deep crawl for ${url}...`);

    let baseUrl;
    try {
      baseUrl = new URL(url).origin;
    } catch (e) {
      return res.status(400).json({ error: "Invalid URL format" });
    }

    const visited = new Set();
    const queue = [url];
    let rawChunks = [];

    const crawler = new CheerioCrawler({
      maxRequestsPerCrawl: 20, // Safe limit for lab
      async requestHandler({ request, $, enqueueLinks, log }) {
        log.info(`Crawling: ${request.url}`);

        // Enqueue links on the same origin
        await enqueueLinks({
          strategy: "same-origin",
        });

        const title = $("title").text().trim() || request.url;
        $(
          'script, style, nav, footer, header, aside, .skip-link, a[href^="#"]',
        ).remove();
        let text = $("body").text().replace(/\s+/g, " ").trim();
        text = text.replace(/Skip to content/gi, "").trim();

        if (text.length > 0) {
          // Recursive-style sentence chunker
          const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
          let currentChunk = "";

          for (let sentence of sentences) {
            sentence = sentence.trim();
            if (!sentence) continue;

            if (
              (currentChunk + " " + sentence).length > 500 &&
              currentChunk.length > 0
            ) {
              rawChunks.push({
                text: currentChunk.trim(),
                url: request.url,
                title: title,
              });
              currentChunk = sentence;
            } else {
              currentChunk += (currentChunk ? " " : "") + sentence;
            }
          }
          if (currentChunk.trim().length > 0) {
            rawChunks.push({
              text: currentChunk.trim(),
              url: request.url,
              title: title,
            });
          }
        }
      },
    });

    console.log(`Starting crawler for ${url}...`);
    await crawler.run([url]);

    console.log(`Embedding ${rawChunks.length} chunks...`);
    let embeddedChunks = [];
    for (let c of rawChunks) {
      try {
        let embeddingRes = await ollama.embeddings({
          model: "nomic-embed-text",
          prompt: `${c.title}\n${c.text}`,
        });
        embeddedChunks.push({
          chunk: c.text,
          source: c.url,
          title: c.title,
          embedding: embeddingRes.embedding,
        });
      } catch (e) {
        console.error(
          "Embedding failed (is nomic-embed-text pulled?)",
          e.message,
        );
        throw new Error(
          "Embedding model failed. Make sure to run: ollama pull nomic-embed-text",
        );
      }
    }

    const index = await readJsonFile(LOCAL_INDEX_FILE);
    index[url] = {
      crawledAt: new Date().toISOString(),
      chunks: embeddedChunks,
    };
    await writeJsonFile(LOCAL_INDEX_FILE, index);

    res.json({
      message: `Successfully crawled site`,
      chunks: embeddedChunks.length,
    });
  } catch (error) {
    console.error(`Failed to deep crawl site: ${error.message}`);
    res.status(500).json({ error: error.message });
  }
});

// --- CHAT HISTORY SERVICE ---
app.get("/api/chats", async (req, res) => {
  const chats = await readJsonFile(CHATS_FILE);
  chats.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
  res.json(chats);
});

app.get("/api/chats/:id", async (req, res) => {
  const chats = await readJsonFile(CHATS_FILE);
  const chat = chats.find((c) => c.id === req.params.id);
  if (!chat) return res.status(404).json({ error: "Chat not found" });
  res.json(chat);
});

app.post("/api/chats/:id", async (req, res) => {
  const { id } = req.params;
  const { messages, targetUrl, model = "llama3.2:3b" } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: "Messages array is required" });
  }

  try {
    let chats = await readJsonFile(CHATS_FILE);
    let chatIndex = chats.findIndex((c) => c.id === id);
    let chat = chatIndex !== -1 ? chats[chatIndex] : null;

    if (!chat) {
      chat = {
        id,
        title: "New Chat",
        messages: [],
        updatedAt: new Date().toISOString(),
      };
      chats.push(chat);
      chatIndex = chats.length - 1;
    }

    if (chat.messages.length === 0 && messages.length > 0) {
      try {
        const titlePrompt = `Generate a very brief, concise title (max 5 words) for this conversation based on the first message:\n\nUser: ${messages[0].content}`;
        const titleRes = await ollama.chat({
          model: model,
          messages: [{ role: "user", content: titlePrompt }],
          stream: false,
        });
        chat.title = titleRes.message.content.replace(/["*]/g, "").trim();
      } catch (e) {
        console.error("Failed to generate title:", e);
      }
    }

    const userQuery = messages[messages.length - 1].content;
    let searchQuery = userQuery;
    let requiresSearch = true;

    // Contextualize query and check search intent
    try {
      let historyText = "";
      if (messages.length >= 3) {
        historyText = `\n\nChat history:\nUser: ${messages[messages.length - 3].content}\nAssistant: ${messages[messages.length - 2].content.slice(0, 300)}...`;
      }

      const currentDate = new Date().toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
      });
      const currentYear = new Date().getFullYear();
      const rewritePrompt = `Rewrite the user's query into a concise search engine query, using the chat history to resolve context if needed. Output ONLY the search query. Today's date is ${currentDate}.
CRITICAL RULE 1: If the user asks about a recurring event or current status without specifying a time frame (e.g., "who won the superbowl", "who is the president"), you MUST append the current year (${currentYear}) to the search query so it retrieves the latest information.
CRITICAL RULE 2: If the user is just saying hello, asking about your capabilities (e.g., "what can you do?", "who are you?"), or making general conversation that doesn't require looking up facts, you MUST output exactly: NO_SEARCH

Example 1:
User's latest query: "who won the superbowl?"
Search query: superbowl winner ${currentYear}

Example 2:
User's latest query: "Hello!"
Search query: NO_SEARCH

Example 3:
User's latest query: "what can you do?"
Search query: NO_SEARCH

Example 4:
Chat history:
User: who won the superbowl?
Assistant: The Seattle Seahawks won Super Bowl LX.
User's latest query: "what about the previous years?"
Search query: superbowl winners previous years

Now do the following:
${historyText ? historyText.trim() + "\n" : ""}User's latest query: "${userQuery}"
Search query:`;

      const rewriteRes = await ollama.chat({
        model: model,
        messages: [{ role: "user", content: rewritePrompt }],
        stream: false,
      });

      const rewritten = rewriteRes.message.content.replace(/["']/g, "").trim();
      if (
        rewritten.toUpperCase() === "NO_SEARCH" ||
        rewritten.toUpperCase() === "IGNORE_SEARCH"
      ) {
        requiresSearch = false;
        searchQuery = "";
        console.log(`Intent check: Query does not require search.`);
      } else if (rewritten && rewritten.length < 150) {
        searchQuery = rewritten;
        console.log(`Rewrote query for search: "${searchQuery}"`);
      }
    } catch (e) {
      console.error(
        "Query rewrite/intent check failed, falling back to original query",
        e,
      );
    }

    // 1. Agentic Search Loop
    let combinedIndices = [];
    let onlineCache = {};
    if (requiresSearch) {
      const localCheck = await readJsonFile(LOCAL_INDEX_FILE);
      onlineCache = await readJsonFile(ONLINE_CACHE_FILE);

      // Load local chunks
      for (const key of Object.keys(localCheck)) {
        if (localCheck[key].chunks) {
          const local = localCheck[key].chunks.map((c) => {
            let actualSource = c.source || key;
            let actualChunk = c.chunk;
            const match = actualChunk.match(/--- Page: (.*?) ---/);
            if (match && match[1]) {
              actualSource = match[1].trim().split(" ")[0];
              actualChunk = actualChunk
                .replace(/--- Page: .*? ---\n/, "")
                .trim();
            }
            return {
              chunk: actualChunk,
              embedding: c.embedding,
              source: actualSource,
              title: c.title,
              isLocal: true,
            };
          });
          combinedIndices = combinedIndices.concat(local);
        }
      }
      // Load online cache
      for (const key of Object.keys(onlineCache)) {
        combinedIndices.push(onlineCache[key]);
      }
    }

    let attempts = 0;
    let maxAttempts = requiresSearch ? 3 : 1;
    let foundAnswer = !requiresSearch;
    let currentSearchQuery = searchQuery;
    let queriesAttempted = requiresSearch && searchQuery ? [searchQuery] : [];
    let selectedChunks = [];
    let topK = [];

    // Memory cleanup for long sessions
    let queryCounter = global.queryCounter || 0;
    queryCounter++;
    global.queryCounter = queryCounter;
    if (queryCounter > 5) {
      console.log("Clearing online cache to save memory...");
      for (const key of Object.keys(onlineCache)) delete onlineCache[key];
      global.queryCounter = 0;
      try {
        await fs.rm("storage", { recursive: true, force: true });
      } catch (e) {}
      await writeJsonFile(ONLINE_CACHE_FILE, onlineCache);
    }

    while (attempts < maxAttempts && !foundAnswer) {
      attempts++;
      let qEmbed = null;
      try {
        qEmbed = await ollama.embeddings({
          model: EMBED_MODEL,
          prompt: currentSearchQuery || userQuery,
        });
      } catch (e) {
        console.error("Embed failed", e);
        break;
      }

      let queryEmbedding = qEmbed.embedding;

      // Score combinedIndices
      for (let item of combinedIndices) {
        let rawSim = cosineSimilarity(queryEmbedding, item.embedding);
        let rankPenalty = item.rank ? item.rank * 0.02 : 0; // Rank penalty for SEO simulation
        item.similarity = rawSim - rankPenalty;
      }
      combinedIndices.sort((a, b) => b.similarity - a.similarity);

      topK = combinedIndices.slice(0, 5);
      let contextText = topK
        .map((c, i) => `[[${i + 1}]] ${c.title || c.source}\n${c.chunk}`)
        .join("\n\n");

      // Evaluate if context has answer
      if (contextText.trim().length > 0) {
        const evalPrompt = `Context:\n${contextText}\n\nQuestion: "${userQuery}"\n\nDoes the context contain enough information to fully answer the question? Reply EXACTLY with 'YES' or 'NO'.`;
        const evalRes = await ollama.chat({
          model: model,
          messages: [{ role: "user", content: evalPrompt }],
          stream: false,
        });
        const evalText = evalRes.message.content.trim().toUpperCase();

        if (evalText.includes("YES")) {
          console.log(
            `Agent loop: Cache/Index contained answer on attempt ${attempts}.`,
          );
          foundAnswer = true;
          break;
        }
      }

      if (attempts < maxAttempts) {
        console.log(
          `Agent loop: Attempt ${attempts} failed. Searching web for: "${currentSearchQuery}"`,
        );

        // Add a small delay between retries to protect students from DuckDuckGo IP bans
        if (attempts > 1) {
          console.log("Sleeping 2.5 seconds to avoid DDG rate limits...");
          await new Promise((r) => setTimeout(r, 2500));
        }

        let webChunks = [];
        try {
          const searchResp = await fetch(
            `https://html.duckduckgo.com/html/?q=${encodeURIComponent(currentSearchQuery)}`,
            { headers: { "User-Agent": "Mozilla/5.0" } },
          );
          const html = await searchResp.text();
          const $ = cheerio.load(html);
          $(".result").each((i, el) => {
            if (i < 5) {
              const snippet = $(el).find(".result__snippet").text().trim();
              const title = $(el).find(".result__a").text().trim();
              let link = $(el).find(".result__a").attr("href");
              if (link && link.includes("uddg="))
                link = decodeURIComponent(link.split("uddg=")[1].split("&")[0]);
              if (snippet)
                webChunks.push({
                  text: snippet,
                  url: link || "Web Search",
                  title: title,
                  rank: i + 1,
                });
            }
          });
        } catch (e) {}

        let urlsToCrawl = webChunks.filter((w) => !onlineCache[w.url]);
        if (urlsToCrawl.length > 0) {
          let crawledContent = {};
          const crawler = new CheerioCrawler({
            maxRequestsPerCrawl: 5,
            maxRequestRetries: 0,
            maxConcurrency: 1, // Crawl 1 page at a time to avoid Cloudflare/WAF IP blocks
            async requestHandler({ request, $ }) {
              $("script, style, nav, footer, header").remove();
              crawledContent[request.url] = $("body")
                .text()
                .replace(/\s+/g, " ")
                .trim()
                .substring(0, 1500);
            },
          });
          let validUrls = [];
          for (const item of urlsToCrawl) {
            try {
              const origin = new URL(item.url).origin;
              const pathname = new URL(item.url).pathname;
              const robText = await (
                await fetch(`${origin}/robots.txt`, {
                  signal: AbortSignal.timeout(2000),
                })
              ).text();

              let allowed = true;
              let userAgentAll = false;
              for (const line of robText.toLowerCase().split("\n")) {
                const trimLine = line.trim();
                if (trimLine.startsWith("user-agent: *")) userAgentAll = true;
                else if (trimLine.startsWith("user-agent:"))
                  userAgentAll = false;
                else if (userAgentAll && trimLine.startsWith("disallow:")) {
                  const disallowPath = trimLine.split(":")[1].trim();
                  if (
                    disallowPath === "/" ||
                    (disallowPath.length > 1 &&
                      pathname.startsWith(disallowPath))
                  ) {
                    allowed = false;
                    break;
                  }
                }
              }

              if (allowed) {
                validUrls.push(item.url);
              } else {
                console.log(`Robots disallowed: ${item.url}`);
              }
            } catch (e) {
              validUrls.push(item.url);
            }
          }
          if (validUrls.length > 0) {
            try {
              await crawler.run(validUrls);
            } catch (e) {}
          }

          for (const item of urlsToCrawl) {
            let textToEmbed =
              crawledContent[item.url] && crawledContent[item.url].length > 50
                ? crawledContent[item.url]
                : item.text;
            try {
              const embedRes = await ollama.embeddings({
                model: EMBED_MODEL,
                prompt: textToEmbed,
              });
              const newChunk = {
                chunk: textToEmbed,
                embedding: embedRes.embedding,
                source: item.url,
                title: item.title,
                isLocal: false,
                rank: item.rank,
              };
              combinedIndices.push(newChunk);
              onlineCache[item.url] = newChunk;
            } catch (e) {}
          }
          await writeJsonFile(ONLINE_CACHE_FILE, onlineCache);
        }

        // Generate new query
        const newQueryPrompt = `The search query "${currentSearchQuery}" did not yield the answer for: "${userQuery}". Generate a completely DIFFERENT and better search query. Output ONLY the exact new search query. Do not add conversational text.

Example 1:
Failed Query: "fifa winners"
User Question: "Who won the fifa world cup in 2018?"
New Search Query: fifa world cup 2018 champion

Now do the following:
Failed Query: "${currentSearchQuery}"
User Question: "${userQuery}"
New Search Query:`;
        const newQueryRes = await ollama.chat({
          model: model,
          messages: [{ role: "user", content: newQueryPrompt }],
          stream: false,
        });
        currentSearchQuery = newQueryRes.message.content
          .replace(/["']/g, "")
          .trim();
        if (currentSearchQuery) queriesAttempted.push(currentSearchQuery);
      }
    }

    // Extract top 5 for LLM
    let localChunks = combinedIndices.filter((c) => c.isLocal);
    let bestLocal = localChunks.length > 0 ? localChunks[0] : null;
    if (bestLocal && !topK.includes(bestLocal)) topK.push(bestLocal);

    selectedChunks = topK.slice(0, 5);
    topK = topK.map((c) => ({
      chunk: c.chunk,
      source: c.source,
      isLocal: c.isLocal,
      similarity: c.similarity || 0,
    }));

    // Prepare Context for Final Generation
    const currentDateStr = new Date().toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
    let systemContent = `You are FakeGPT, a highly descriptive, friendly, and engaging AI chatbot. Your knowledge cutoff is December 2023, but you are equipped with powerful agentic tools to overcome this! Today's date is ${currentDateStr}.

YOUR CAPABILITIES:
- You have an autonomous Agentic Search Loop that can query a real-time web search engine to find up-to-date information.
- You can crawl and read full web pages to deeply understand topics.
If the user asks what you can do, proudly describe these capabilities!

You have been provided with 'Context' retrieved from your search engine. 
CRITICAL RULE: For any factual question, you MUST base your answer SOLELY on the provided Context. DO NOT use your internal general knowledge to answer factual questions. If the provided Context does not contain the answer, you must state that you couldn't find the information in the current search results.
DO NOT say "Based on the context" or "In the text you shared". Just answer naturally as if you found it online.

CRITICAL: At the end of EVERY response, ask a friendly follow-up question to keep the conversation engaging.`;
    if (selectedChunks.length > 0) {
      let contextText = "";
      selectedChunks.forEach((c, idx) => {
        let header = `SOURCE [[${idx + 1}]] URL: ${c.source}`;
        if (c.title) header += ` TITLE: ${c.title}`;
        contextText += `${header}\n${c.chunk}\n\n`;
      });
      systemContent += `\n\nContext:\n${contextText}\n\nCRITICAL CITATION RULES:
1. You MUST add a citation at the end of EVERY factual sentence you write.
2. Format your citations EXACTLY like this example: "Spain won the tournament in 2026 [[1]](https://worldcupwiki.com)."
3. The citation MUST be a markdown link using the SOURCE number and URL provided above. Do NOT write the website name.
4. If a sentence uses information from multiple sources, cite them all: [[1]](URL) [[2]](URL).
5. Do NOT generate a reference list at the end of your response. ONLY use inline citations.
6. If the provided Context does not contain the answer, you MUST say 'I cannot find the answer in the current search results' instead of using your own knowledge.`;
    }

    const ollamaMessages = [
      { role: "system", content: systemContent },
      ...messages.map((m) => ({ role: m.role, content: m.content })),
    ];

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    let fullContent = "";

    const stream = await ollama.chat({
      model: model,
      messages: ollamaMessages,
      stream: true,
    });

    for await (const chunk of stream) {
      if (chunk.message.content) {
        fullContent += chunk.message.content;
        res.write(
          `data: ${JSON.stringify({ content: chunk.message.content })}\n\n`,
        );
      }
    }

    if (topK && topK.length > 0) {
      res.write(
        `data: ${JSON.stringify({ sources: topK, rewrittenQuery: queriesAttempted.length > 0 ? queriesAttempted.join(" ➔ ") : searchQuery })}\n\n`,
      );
    } else {
      res.write(
        `data: ${JSON.stringify({ sources: [], rewrittenQuery: queriesAttempted.length > 0 ? queriesAttempted.join(" ➔ ") : searchQuery })}\n\n`,
      );
    }

    res.write("data: [DONE]\n\n");
    res.end();

    const newMessages = [
      ...messages,
      {
        role: "assistant",
        content: fullContent,
        sources: topK || [],
        rewrittenQuery: queriesAttempted.length > 0 ? queriesAttempted.join(" ➔ ") : searchQuery,
      },
    ];
    chat.messages = newMessages;
    chat.updatedAt = new Date().toISOString();
    chats[chatIndex] = chat;

    await writeJsonFile(CHATS_FILE, chats);
  } catch (error) {
    console.error("Chat error:", error);
    if (!res.writableEnded) {
      res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
      res.end();
    }
  }
});

// Clear Data Routes
app.post("/api/clear-history", async (req, res) => {
  try {
    await writeJsonFile(CHATS_FILE, []);
    res.json({ success: true });
  } catch (error) {
    console.error("Clear history error:", error);
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/clear-index", async (req, res) => {
  try {
    await writeJsonFile(LOCAL_INDEX_FILE, {});
    res.json({ success: true });
  } catch (error) {
    console.error("Clear index error:", error);
    res.status(500).json({ error: error.message });
  }
});

// Catch-all route to serve the React SPA for any unmatched paths
app.use((req, res, next) => {
  if (req.method === "GET" && !req.path.startsWith("/api")) {
    res.sendFile(path.join(FRONTEND_DIR, "index.html"));
  } else {
    next();
  }
});

const PORT = process.env.PORT || 3002;
app.listen(PORT, () => {
  console.log(`FakeGPT backend running on port ${PORT}`);
});
