// 1. Agentic Search Loop
let combinedIndices = [];
const localCheck = await readJsonFile(LOCAL_INDEX_FILE);
const onlineCache = await readJsonFile(ONLINE_CACHE_FILE);

// Load local chunks
for (const key of Object.keys(localCheck)) {
  if (localCheck[key].chunks) {
    const local = localCheck[key].chunks.map((c) => {
      let actualSource = c.source || key;
      let actualChunk = c.chunk;
      const match = actualChunk.match(/--- Page: (.*?) ---/);
      if (match && match[1]) {
        actualSource = match[1].trim().split(" ")[0];
        actualChunk = actualChunk.replace(/--- Page: .*? ---\n/, "").trim();
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

let attempts = 0;
let maxAttempts = requiresSearch ? 3 : 1;
let foundAnswer = !requiresSearch;
let currentSearchQuery = searchQuery;
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
    item.similarity = cosineSimilarity(queryEmbedding, item.embedding);
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
            });
        }
      });
    } catch (e) {}

    let urlsToCrawl = webChunks.filter((w) => !onlineCache[w.url]);
    if (urlsToCrawl.length > 0) {
      let crawledContent = {};
      const crawler = new CheerioCrawler({
        maxRequestsPerCrawl: 5,
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
          const robText = await (
            await fetch(`${origin}/robots.txt`, {
              signal: AbortSignal.timeout(2000),
            })
          ).text();
          if (
            !robText.toLowerCase().includes("disallow: /\n") &&
            !robText.toLowerCase().includes("disallow: /\r")
          ) {
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
          };
          combinedIndices.push(newChunk);
          onlineCache[item.url] = newChunk;
        } catch (e) {}
      }
      await writeJsonFile(ONLINE_CACHE_FILE, onlineCache);
    }

    // Generate new query
    const newQueryPrompt = `The search query "${currentSearchQuery}" did not yield the answer for: "${userQuery}". Generate a completely DIFFERENT and better search query. Output ONLY the new search query.`;
    const newQueryRes = await ollama.chat({
      model: model,
      messages: [{ role: "user", content: newQueryPrompt }],
      stream: false,
    });
    currentSearchQuery = newQueryRes.message.content
      .replace(/["']/g, "")
      .trim();
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
  similarity: c.similarity,
}));
