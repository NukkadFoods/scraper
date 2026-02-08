"""
FastAPI Backend for Google News Image Scraper
Uses Groq LLM for intelligent metadata-based filtering (no heavy ML models).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus, urlparse
import urllib.parse

# Load .env for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required in production

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

# Response folder path
RESPONSE_DIR = Path(__file__).parent / "response"

# Rotating User-Agents to avoid rate limiting on datacenter IPs
USER_AGENTS = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    # Chrome on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0",
    # Firefox on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:132.0) Gecko/20100101 Firefox/132.0",
    # Chrome on Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    # Safari on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
]


def get_random_user_agent() -> str:
    """Get a random browser User-Agent to avoid rate limiting."""
    return random.choice(USER_AGENTS)

# Known logo/icon hashes to block
BLOCKED_HASHES = {"8c30b0eaece8e454"}  # Google News logo

# Groq API configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("groqapikey", "")
GROQ_MODEL = "llama-3.1-8b-instant"  # Fast and free

# Cache for query analysis (avoid repeated API calls)
_query_cache: dict[str, dict] = {}


async def analyze_query_with_groq(query: str, client: httpx.AsyncClient) -> dict:
    """
    Use Groq to analyze the query and extract search intelligence.
    Returns: {
        "type": "person" | "topic" | "object" | "event",
        "canonical_name": str,  # Full proper name if person
        "aliases": list[str],   # Alternative names/spellings
        "keywords": list[str],  # Related terms to look for
        "negative_keywords": list[str],  # Terms that indicate wrong match
        "description": str      # Brief description for context
    }
    """
    cache_key = query.lower().strip()
    if cache_key in _query_cache:
        return _query_cache[cache_key]

    if not GROQ_API_KEY:
        # Fallback without API
        return {
            "type": "unknown",
            "canonical_name": query,
            "aliases": [query.lower()],
            "keywords": query.lower().split(),
            "negative_keywords": [],
            "description": ""
        }

    prompt = f"""Analyze this search query for finding images: "{query}"

Return a SINGLE JSON object (not an array) with these fields:
- type: "person", "topic", "object", or "event"
- canonical_name: Full proper name (e.g., "Donald Trump" for "trump")
- aliases: List of alternative names/spellings to match in text
- keywords: Related terms that suggest relevance
- negative_keywords: Terms that indicate a DIFFERENT subject (for disambiguation)
- description: One-line description

Example for "modi":
{{"type":"person","canonical_name":"Narendra Modi","aliases":["Modi","PM Modi","Narendra Damodardas Modi"],"keywords":["india","prime minister","bjp"],"negative_keywords":["lalit modi","nirav modi"],"description":"Prime Minister of India"}}

Return ONLY the JSON object, no markdown or explanation."""

    try:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 250
            },
            timeout=10.0
        )

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            # Parse JSON from response
            content = content.strip()
            if content.startswith("```"):
                parts = content.split("```")
                content = parts[1] if len(parts) > 1 else parts[0]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            result = json.loads(content)

            # Handle if LLM returns a list instead of object
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            if isinstance(result, dict):
                _query_cache[cache_key] = result
                print(f"Groq analysis: {query} -> type={result.get('type')}, canonical={result.get('canonical_name')}")
                return result
    except Exception as e:
        print(f"Groq query analysis error: {e}")

    # Fallback: basic text matching
    fallback = {
        "type": "unknown",
        "canonical_name": query,
        "aliases": [query.lower(), query.title(), query.upper()],
        "keywords": query.lower().split(),
        "negative_keywords": [],
        "description": ""
    }
    _query_cache[cache_key] = fallback
    return fallback


def check_text_relevance(
    context_text: str,
    query_info: dict,
    query: str
) -> tuple[bool, float, bool, dict]:
    """
    Check if image context text indicates relevance to the query.
    Uses intelligent text matching with Groq-provided query intelligence.

    Returns: (is_relevant, confidence_score, is_solo, debug_scores)
    """
    context_lower = context_text.lower()
    query_lower = query.lower()

    canonical = query_info.get("canonical_name", query).lower()
    aliases = [a.lower() for a in query_info.get("aliases", [query])]
    keywords = [k.lower() for k in query_info.get("keywords", [])]
    negative_keywords = [n.lower() for n in query_info.get("negative_keywords", [])]
    query_type = query_info.get("type", "unknown")

    scores = {
        "exact_match": 0,
        "alias_match": 0,
        "keyword_match": 0,
        "negative_match": 0,
        "query_type": query_type
    }

    # 1. Exact canonical name match (highest weight)
    if canonical in context_lower:
        scores["exact_match"] = 1.0

    # 2. Alias matches
    alias_matches = sum(1 for alias in aliases if alias in context_lower)
    if alias_matches > 0:
        scores["alias_match"] = min(1.0, alias_matches * 0.4)

    # 3. Keyword matches (supporting evidence)
    keyword_matches = sum(1 for kw in keywords if kw in context_lower and len(kw) > 2)
    if keyword_matches > 0:
        scores["keyword_match"] = min(0.5, keyword_matches * 0.15)

    # 4. Negative keyword check (reduces confidence)
    negative_matches = sum(1 for neg in negative_keywords if neg in context_lower)
    if negative_matches > 0:
        scores["negative_match"] = min(0.8, negative_matches * 0.3)

    # Calculate final score
    positive_score = scores["exact_match"] + scores["alias_match"] + scores["keyword_match"]
    negative_penalty = scores["negative_match"]

    # For person queries, check for group indicators
    is_solo = True
    if query_type == "person":
        group_indicators = ["with", "meets", "and", "alongside", "together", "group", "team", "family"]
        if any(ind in context_lower for ind in group_indicators):
            # Check if the query subject is the focus despite group context
            # e.g., "Trump meets Biden" - Trump is still the subject
            words_before_indicator = context_lower.split(canonical)[0] if canonical in context_lower else ""
            if len(words_before_indicator) < 20:  # Subject is early in text = likely the focus
                is_solo = True
            else:
                is_solo = False
                scores["group_indicator"] = True

    final_score = max(0, min(1.0, positive_score - negative_penalty))

    # Relevance threshold
    is_relevant = final_score >= 0.3 or (scores["exact_match"] > 0)

    # If we have strong negative matches and weak positive, reject
    if negative_penalty > positive_score and scores["exact_match"] == 0:
        is_relevant = False

    return is_relevant, final_score, is_solo, scores

app = FastAPI(
    title="Google News Image Scraper API",
    description="Scrape relevant images from Google News using Groq-powered intelligent filtering",
    version="3.0.0",
)

@app.on_event("startup")
def startup_event():
    """Initialize on startup."""
    if GROQ_API_KEY:
        print(f"Groq API configured (model: {GROQ_MODEL})")
    else:
        print("WARNING: No Groq API key found. Using basic text matching.")
    print("Image scraper ready (lightweight mode - no ML models).")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Keep-alive self-ping to prevent cold starts on Render
KEEP_ALIVE_INTERVAL = 13 * 60  # 13 minutes in seconds

async def keep_alive_ping():
    """Background task to ping ourselves every 13 minutes to prevent cold starts."""
    # Get the service URL from environment (Render sets RENDER_EXTERNAL_URL)
    service_url = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000")
    health_url = f"{service_url}/health"
    
    while True:
        await asyncio.sleep(KEEP_ALIVE_INTERVAL)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(health_url, timeout=30)
                print(f"Keep-alive ping: {response.status_code}")
        except Exception as e:
            print(f"Keep-alive ping failed: {e}")


@app.on_event("startup")
async def startup_event():
    """Start background tasks on app startup."""
    # Only run keep-alive in production (when RENDER_EXTERNAL_URL is set)
    if os.environ.get("RENDER_EXTERNAL_URL"):
        asyncio.create_task(keep_alive_ping())
        print("Keep-alive background task started")


class ImageItem(BaseModel):
    """Single image in the response."""
    url: str = Field(..., description="Original source URL of the image")
    width: Optional[int] = Field(None, description="Image width in pixels")
    height: Optional[int] = Field(None, description="Image height in pixels")
    size_bytes: Optional[int] = Field(None, description="File size in bytes")
    base64: Optional[str] = Field(None, description="Base64 encoded image data")
    format: Optional[str] = Field(None, description="Image format (jpg, png, etc)") # MODIFIED
    saved_path: Optional[str] = Field(None, description="Local path where image is saved")
    relevance_score: float = Field(..., description="Text relevance score (0-1)")
    is_solo: bool = Field(True, description="Whether the image shows a single person")
    debug_scores: Optional[dict] = Field(None, description="Detailed relevance scores for debugging")


class ScrapeResponse(BaseModel):
    """Response from the scrape endpoint."""
    success: bool
    query: str
    requested_count: int = Field(..., description="Number of images requested")
    total_found: int = Field(..., description="Total image URLs found on page")
    total_processed: int = Field(..., description="Images processed before reaching target")
    total_returned: int = Field(..., description="Unique images returned")
    images: list[ImageItem]
    response_folder: Optional[str] = None
    error: Optional[str] = None


def compute_dhash(image_bytes: bytes, hash_size: int = 8) -> str | None:
    """Compute perceptual difference hash for deduplication."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            image = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            diff = []
            for row in range(hash_size):
                for col in range(hash_size):
                    pixel_left = image.getpixel((col, row))
                    pixel_right = image.getpixel((col + 1, row))
                    diff.append(pixel_left > pixel_right)
            
            decimal_value = 0
            hex_string = []
            for index, value in enumerate(diff):
                if value:
                    decimal_value += 2**(index % 8)
                if (index % 8) == 7:
                    hex_string.append(hex(decimal_value)[2:].rjust(2, '0'))
                    decimal_value = 0
            return "".join(hex_string)
    except Exception:
        return None


def hamming_distance(h1: str, h2: str) -> int:
    """Compute Hamming distance between two hashes."""
    if not h1 or not h2 or len(h1) != len(h2):
        return 999
    try:
        return bin(int(h1, 16) ^ int(h2, 16)).count('1')
    except ValueError:
        return 999


def extract_size_from_url(url: str) -> tuple[int, int]:
    """Extract width/height hints from Google News image URLs."""
    # Google URLs often have -w400-h224 format
    w_match = re.search(r'-w(\d+)', url)
    h_match = re.search(r'-h(\d+)', url)
    width = int(w_match.group(1)) if w_match else 0
    height = int(h_match.group(1)) if h_match else 0
    return width, height


def upscale_google_news_url(url: str, target_width: int = 1200) -> str:
    """
    Transform Google News image URLs to request larger versions.
    Google News URLs have format: ...=-w200-h112-p-df-rw
    We can change the width/height to get larger images.
    """
    if "news.google.com" not in url:
        return url
    
    # Pattern: -w{num}-h{num}
    # Replace with larger dimensions, maintaining aspect ratio
    w_match = re.search(r'-w(\d+)', url)
    h_match = re.search(r'-h(\d+)', url)
    
    if w_match and h_match:
        orig_w = int(w_match.group(1))
        orig_h = int(h_match.group(1))
        
        if orig_w > 0:
            scale = target_width / orig_w
            new_w = target_width
            new_h = int(orig_h * scale)
            
            # Replace in URL
            url = re.sub(r'-w\d+', f'-w{new_w}', url)
            url = re.sub(r'-h\d+', f'-h{new_h}', url)
    
    return url


async def fetch_duckduckgo_images(query: str, client: httpx.AsyncClient) -> tuple[str, str]:
    """
    Fallback: Fetch images from DuckDuckGo (more lenient with datacenter IPs).
    Returns (html_content, final_url).
    """
    # DuckDuckGo image search URL
    url = f"https://duckduckgo.com/?q={quote_plus(query)}&iax=images&ia=images"

    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://duckduckgo.com/",
    }

    response = await client.get(url, headers=headers)
    return response.text, str(response.url)


def extract_image_urls(html: str, base_url: str, query: str = "") -> list[dict]:
    """
    Extract all image URLs from HTML with context for relevance filtering.
    Returns images sorted by: relevance to query, then by size.
    """
    soup = BeautifulSoup(html, "html.parser")
    images = []
    seen_urls = set()
    query_lower = query.lower().strip()
    query_words = set(query_lower.split())
    
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if not src or src in seen_urls:
            continue
        
        # Skip data URIs and tiny icons
        if src.startswith("data:"):
            continue
            
        # Make absolute URL
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            parsed = urlparse(base_url)
            src = f"{parsed.scheme}://{parsed.netloc}{src}"
        elif not src.startswith("http"):
            continue
        
        if src in seen_urls:
            continue
        seen_urls.add(src)
        
        # Extract size hints
        width, height = extract_size_from_url(src)
        
        # Also check HTML attributes
        if not width:
            width = int(img.get("width", 0) or 0)
        if not height:
            height = int(img.get("height", 0) or 0)
        
        # === CONTEXT EXTRACTION ===
        # Get alt text
        alt_text = (img.get("alt") or "").lower()
        
        # Get title attribute
        title_text = (img.get("title") or "").lower()
        
        # Find parent article/container and extract headline
        context_text = ""
        parent = img.parent
        for _ in range(10):  # Walk up to 10 levels (reverted to find broad context)
            if parent is None:
                break
            # Look for article, heading, or text content
            if parent.name in ["article", "div", "a", "figure"]:
                # Find headings
                heading = parent.find(["h1", "h2", "h3", "h4", "h5", "h6"])
                if heading and heading.get_text().strip():
                    context_text += f" [HEAD]: {heading.get_text().strip().lower()}"
                
                # Find figcaption
                caption = parent.find("figcaption")
                if caption and caption.get_text().strip():
                    context_text += f" [CAP]: {caption.get_text().strip().lower()}"
                
                # Get aria-label
                aria = parent.get("aria-label", "")
                if aria and aria.strip():
                    context_text += f" [ARIA]: {aria.strip().lower()}"
                
            parent = parent.parent
        
        # Separate direct context (specific to image) from surrounding context
        direct_context = f"{alt_text} {title_text}".strip()
        surrounding_context = context_text.strip()
        
        # === RELEVANCE SCORING ===
        relevance_score = 0
        query_words = query_lower.split()
        
        # 0. Format Boost (WebP is often main article image on Google News)
        if "webp" in src.lower() or src.endswith("-rw"):  # Google News often uses -rw for WebP
            relevance_score += 50
        
        # 1. Direct Match (Alt/Title) - Highest Priority
        if query_lower in direct_context:
            relevance_score += 1000
        elif any(word in direct_context for word in query_words if len(word) > 3):
            relevance_score += 500
            
        # 2. Check clear Caption/Figcaption (often reliable)
        caption = img.find_parent("figure")
        if caption:
            caption_text = caption.get_text().lower()
            if query_lower in caption_text:
                relevance_score += 800
            elif any(word in caption_text for word in query_words if len(word) > 3):
                relevance_score += 400
        
        # 3. Surrounding Context (Headlines, Article) - Lower Priority
        if query_lower in surrounding_context:
            relevance_score += 100
        elif any(word in surrounding_context for word in query_words if len(word) > 3):
            relevance_score += 30
            
        images.append({
            "src": src,
            "estimated_width": width,
            "estimated_height": height,
            "area": width * height,
            "context": f"{direct_context} {surrounding_context}"[:500],  # Truncate for memory
            "relevance_score": relevance_score,
        })
    
    # Sort by relevance first, then by area (largest first)
    images.sort(key=lambda x: (x["relevance_score"], x["area"]), reverse=True)
    
    return images


async def download_and_validate_image(
    client: httpx.AsyncClient,
    url: str,
    min_width: int,
    min_height: int,
    seen_hashes: set[str],
    allowed_formats: list[str] | None,
) -> dict | None:
    """
    Download an image, validate it, and check for duplicates.
    Returns image data dict if valid and unique, None otherwise.
    """
    try:
        response = await client.get(
            url,
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": get_random_user_agent()},
        )
        
        if response.status_code != 200:
            return None
        
        content = response.content
        content_type = response.headers.get("content-type", "").lower()
        
        # Skip HTML responses
        if "text/html" in content_type:
            return None
        
        # Validate image format
        is_png = content[:4] == b'\x89PNG'
        is_jpeg = content[:2] == b'\xff\xd8'
        is_webp = content[:4] == b'RIFF' and b'WEBP' in content[:12]
        is_gif = content[:6] in (b'GIF87a', b'GIF89a')
        
        if not (is_png or is_jpeg or is_webp or is_gif):
            return None
        
        # Determine format
        if is_png:
            fmt = "png"
        elif is_jpeg:
            fmt = "jpg"
        elif is_webp:
            fmt = "webp"
        elif is_gif:
            fmt = "gif"
        else:
            fmt = "jpg"
        
        # Check allowed formats
        if allowed_formats and fmt not in allowed_formats and not (fmt == "jpg" and "jpeg" in allowed_formats):
            return None
        
        # Get actual dimensions
        try:
            with Image.open(io.BytesIO(content)) as img:
                width, height = img.size
        except Exception:
            return None
        
        # Check minimum size
        if width < min_width or height < min_height:
            return None
        
        # Compute perceptual hash for deduplication
        phash = compute_dhash(content)
        if not phash:
            return None
        
        # Check against blocked hashes (logos, icons)
        for blocked in BLOCKED_HASHES:
            if hamming_distance(phash, blocked) <= 4:
                return None
        
        # Check for duplicates
        for seen in seen_hashes:
            if hamming_distance(phash, seen) <= 4:
                return None
        
        # This is a valid, unique image!
        seen_hashes.add(phash)
        
        return {
            "content": content,
            "width": width,
            "height": height,
            "size_bytes": len(content),
            "format": fmt,
            "url": url,
        }
        
    except Exception:
        return None


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "Google News Image Scraper API v2.0 - Groq-powered filtering"}


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for keep-alive pings."""
    return {"status": "healthy", "timestamp": __import__("datetime").datetime.now().isoformat()}

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check system state."""
    import sys
    import pkg_resources
    return {
        "python": sys.version,
        "packages": [f"{p.key}=={p.version}" for p in pkg_resources.working_set],
        "env": {k: v for k, v in os.environ.items() if "KEY" not in k and "TOKEN" not in k}
    }


async def process_candidate(
    img_data: dict,
    client: httpx.AsyncClient,
    q: str,
    query_info: dict,
    strict: bool,
    solo_only: bool,
    min_width: int,
    min_height: int,
    seen_hashes: set[str],
    allowed_formats: list[str] | None,
) -> dict | None:
    """
    Process a single image candidate. Returns dict with image data (not saved yet).
    Saving happens later after we select the top results.
    """
    context_text = img_data.get("context", "")

    # Text-based relevance check (fast, no ML)
    is_relevant, text_score, is_solo, scores_dict = check_text_relevance(
        context_text, query_info, q
    )

    # Strict mode: require text relevance before downloading
    if strict and not is_relevant:
        return None

    # Solo-only mode for person queries
    if solo_only and query_info.get("type") == "person" and not is_solo:
        if scores_dict.get("exact_match", 0) < 0.8:
            return None

    # Download the image
    result = await download_and_validate_image(
        client,
        upscale_google_news_url(img_data["src"]),
        min_width,
        min_height,
        seen_hashes,
        allowed_formats,
    )

    if not result:
        return None

    # Prepare debug scores
    scores_dict["metadata_relevance"] = img_data.get("relevance_score", 0)
    scores_dict["context_snippet"] = context_text[:150]

    # Return data dict (don't save yet)
    return {
        "url": img_data["src"],
        "width": result["width"],
        "height": result["height"],
        "content": result["content"],  # Keep in memory for now
        "format": result["format"],
        "relevance_score": text_score,
        "is_solo": is_solo,
        "debug_scores": scores_dict,
    }

@app.get("/scrape", response_model=ScrapeResponse, tags=["Scraping"])
async def scrape_images(
    q: str = Query(..., min_length=1, max_length=200, description="Search term"),
    count: int = Query(default=10, ge=1, le=100, description="Number of unique images to return"),
    min_width: int = Query(default=100, ge=50, le=2000, description="Minimum image width"),
    min_height: int = Query(default=100, ge=50, le=2000, description="Minimum image height"),
    include_base64: bool = Query(default=False, description="Include base64 data"),
    formats: Optional[str] = Query(default=None, description="Allowed formats: jpg,png,webp,gif"),
    strict: bool = Query(default=True, description="Only return images that pass text relevance check"),
    solo_only: bool = Query(default=True, description="For person queries, only return images of the person alone (not in groups)"),
):
    """
    Scrape exactly `count` unique images from Google News.

    Uses Groq LLM for intelligent query understanding and text-based filtering.
    No heavy ML models - fast and lightweight for deployment.
    """
    try:
        # Parse formats
        allowed_formats = None
        if formats:
            allowed_formats = [f.strip().lower() for f in formats.split(",")]
            valid = {"jpg", "jpeg", "png", "webp", "gif"}
            allowed_formats = [f for f in allowed_formats if f in valid] or None

        # Create response folder
        safe_name = re.sub(r'[^\w\s-]', '', q).strip().replace(' ', '_').lower()
        if not safe_name:
            safe_name = f"query_{hashlib.md5(q.encode()).hexdigest()[:8]}"

        response_folder = RESPONSE_DIR / safe_name
        response_folder.mkdir(parents=True, exist_ok=True)

        # Clear existing images
        for ext in ["jpg", "png", "webp", "gif", "jpeg"]:
            for f in response_folder.glob(f"*.{ext}"):
                f.unlink(missing_ok=True)

        # Build Google News URL
        url = f"https://news.google.com/search?q={quote_plus(q)}"

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            # Step 1: Analyze query with Groq (fast, cached)
            query_info = await analyze_query_with_groq(q, client)
            print(f"Query '{q}' analyzed: type={query_info.get('type')}, aliases={query_info.get('aliases', [])[:3]}")

            # Step 2: Fetch Google News page with retry logic
            max_retries = 3
            response = None

            for attempt in range(max_retries):
                # Add small random delay to appear more human-like
                if attempt > 0:
                    await asyncio.sleep(random.uniform(1, 3))

                # Full browser-like headers
                headers = {
                    "User-Agent": get_random_user_agent(),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1",
                    "Cache-Control": "max-age=0",
                }

                response = await client.get(url, headers=headers)

                if response.status_code == 200:
                    break
                elif response.status_code == 429:
                    print(f"Rate limited (attempt {attempt + 1}/{max_retries}), retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                else:
                    break

            if response.status_code != 200:
                # Fallback to DuckDuckGo if Google is blocked
                print(f"Google News blocked ({response.status_code}), trying DuckDuckGo...")
                try:
                    html_content, final_url = await fetch_duckduckgo_images(q, client)
                except Exception as ddg_err:
                    raise HTTPException(status_code=502, detail=f"Both Google ({response.status_code}) and DuckDuckGo failed")
            else:
                html_content = response.text
                final_url = str(response.url)

        # Extract all image URLs with context and relevance scoring
        image_urls = extract_image_urls(html_content, final_url, query=q)
        total_found = len(image_urls)

        # Parallel Processing with text-based filtering (no heavy ML)
        candidates: list[dict] = []
        target_buffer_size = count * 3  # Process more candidates since text filtering is fast
        seen_hashes: set[str] = set()

        sem = asyncio.Semaphore(15)  # Higher concurrency since text filtering is fast

        async with httpx.AsyncClient() as client:
            async def semaphore_task(img_data):
                async with sem:
                    return await process_candidate(
                        img_data, client, q, query_info, strict, solo_only,
                        min_width, min_height, seen_hashes, allowed_formats
                    )

            tasks = []
            for img_data in image_urls[:target_buffer_size]:
                tasks.append(semaphore_task(img_data))

            # Run all tasks
            results = await asyncio.gather(*tasks)

            # Filter None (failed/skipped)
            candidates = [r for r in results if r is not None]

        # Sort candidates by relevance
        def candidate_sort_key(item: dict):
            meta_score = item.get("debug_scores", {}).get("metadata_relevance", 0)
            exact_match = item.get("debug_scores", {}).get("exact_match", 0)
            is_webp = 1 if item.get("format") == "webp" else 0
            return (exact_match, meta_score, is_webp, item.get("relevance_score", 0))

        candidates.sort(key=candidate_sort_key, reverse=True)

        # Homogeneity Filter: prefer WebP images (often main article images on Google News)
        if strict and any(c.get("format") == "webp" for c in candidates):
            webp_candidates = [c for c in candidates if c.get("format") == "webp"]
            if len(webp_candidates) >= count:
                candidates = webp_candidates

        # Take top `count` and save ONLY those to disk
        final_candidates = candidates[:count]
        images: list[ImageItem] = []

        for i, cand in enumerate(final_candidates):
            filename = f"image_{i + 1:03d}.{cand['format']}"
            filepath = response_folder / filename
            with open(filepath, "wb") as f:
                f.write(cand["content"])

            # Convert to base64 if requested
            b64_str = None
            if include_base64:
                b64_str = base64.b64encode(cand["content"]).decode("utf-8")

            images.append(ImageItem(
                url=cand["url"],
                width=cand["width"],
                height=cand["height"],
                size_bytes=len(cand["content"]),
                base64=b64_str,
                format=cand["format"],
                saved_path=str(filepath),
                relevance_score=float(cand["relevance_score"]),
                is_solo=cand["is_solo"],
                debug_scores=cand["debug_scores"],
            ))

        return ScrapeResponse(
            success=True,
            query=q,
            requested_count=count,
            total_found=total_found,
            total_processed=len(tasks),
            total_returned=len(images),
            images=images,
            response_folder=str(response_folder),
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


@app.get("/scrape/base64", tags=["Scraping"])
async def scrape_images_base64(
    q: str = Query(..., min_length=1, max_length=200),
    count: int = Query(default=10, ge=1, le=50),
):
    """Convenience endpoint with base64 included."""
    return await scrape_images(q=q, count=count, include_base64=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
