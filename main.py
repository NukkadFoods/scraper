import asyncio
import sys
import json
from pathlib import Path

# Import scanner
try:
    from media_scrape import media_scrape
except ImportError:
    # If running from parent dir
    sys.path.append(str(Path(__file__).parent))
    from media_scrape import media_scrape

async def main():
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "trump golden statue"
        
    print(f"Scraping Google News for: '{query}'")
    url = f"https://news.google.com/search?q={query.replace(' ', '+')}"
    output_dir = Path.cwd() / "downloads" / query.replace(" ", "_")
    
    print(f"URL: {url}")
    print(f"Output: {output_dir}")
    
    result = await media_scrape(
        url=url,
        output_dir=str(output_dir),
        allowed_formats=["webp"],
        deduplicate=True,
        max_items=50
    )
    
    print("\n--- Results ---")
    print(f"Total Found: {result.get('total_found')}")
    print(f"Downloaded: {result.get('total_downloaded')}")
    
    # Save metadata
    if result.get('output_dir'):
        meta_file = Path(result['output_dir']) / "metadata.json"
        meta_file.write_text(json.dumps(result, default=str, indent=2))
        print(f"Metadata saved to: {meta_file}")

if __name__ == "__main__":
    asyncio.run(main())
