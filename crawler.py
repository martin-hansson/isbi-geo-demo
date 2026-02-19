import asyncio
import sys
import os
import csv
from crawl4ai import AsyncWebCrawler

async def crawl_from_csv(csv_filename):
    # 1. Ensure the data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")

    urls = []
    
    # 2. Read the CSV file
    try:
        with open(csv_filename, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # This assumes your CSV has a header named 'url'
                if 'url' in row:
                    urls.append(row['url'])
                else:
                    print(f"Skipping row: No 'url' column found in {csv_filename}")
    except FileNotFoundError:
        print(f"Error: The file '{csv_filename}' was not found.")
        return

    if not urls:
        print("No URLs found to crawl.")
        return

    # 3. Run the Crawler
    async with AsyncWebCrawler(verbose=True) as crawler:
        results = await crawler.arun_many(urls)
        
        for result in results:
            if result.success:
                # Create a filename-safe string from the URL
                safe_name = result.url.replace("https://", "").replace("/", "_").strip("_")
                with open(f"data/{safe_name}.md", "w", encoding="utf-8") as f:
                    f.write(result.markdown)
                print(f"Saved: {result.url}")
            else:
                print(f"Failed to crawl: {result.url}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python crawler.py <your_file.csv>")
    else:
        target_file = sys.argv[1]
        asyncio.run(crawl_from_csv(target_file))