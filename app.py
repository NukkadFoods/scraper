"""
FastAPI Backend for Google News Image Scraper
Implements streaming approach with CLIP-based semantic filtering.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import os
import re
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus, urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

# CLIP for semantic image filtering
from onnx_clip import OnnxClip

# Response folder path
RESPONSE_DIR = Path(__file__).parent / "response"

# Browser User-Agent
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# Known logo/icon hashes to block
BLOCKED_HASHES = {"8c30b0eaece8e454"}  # Google News logo

# Initialize CLIP model (loads once on startup)
_clip_model: OnnxClip | None = None

def get_clip_model() -> OnnxClip:
    """Lazy-load CLIP model on first use."""
    global _clip_model
    if _clip_model is None:
        _clip_model = OnnxClip(batch_size=1)
    return _clip_model


def check_image_relevance(
    image_bytes: bytes, 
    query: str, 
    threshold: float = 0.25,
    solo_only: bool = True
) -> tuple[bool, float, bool]:
    """
    Use CLIP to check if an image shows the queried subject (person/thing).
    
    Args:
        image_bytes: Raw image data
        query: Search query (e.g., "trump")
        threshold: Minimum relevance score (0-1)
        solo_only: If True, reject group photos for person queries
    
    Returns (is_relevant, confidence_score, is_solo)
    """
    try:
        clip = get_clip_model()
        import numpy as np
        
        # Open and resize image for faster processing
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Resize large images to speed up CLIP (CLIP uses 224x224 anyway)
        if max(image.size) > 512:
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        
        # Check if this is likely a person query
        is_person_query = len(query.split()) <= 2 and not any(
            word in query.lower() for word in ["landscape", "building", "car", "food", "animal", "house", "city"]
        )
        
        if is_person_query:
            # Very specific prompts to detect person's FACE
            labels = [
                f"a close-up photo showing {query}'s face clearly visible",
                f"a photo of {query} standing with other people in a group",
                "a photo of a building, landscape, or scene without any person's face",
                "a photo of someone else, a different person",
            ]
        else:
            # General object/thing query
            labels = [
                f"a photo of {query}",
                "a photo of something completely different and unrelated",
            ]
        
        # Get embeddings
        image_embedding = clip.get_image_embeddings([image])
        text_embeddings = clip.get_text_embeddings(labels)
        
        # Calculate similarities
        similarities = (image_embedding @ text_embeddings.T).flatten()
        
        # Softmax to get probabilities
        exp_sims = np.exp(similarities - np.max(similarities))
        probs = exp_sims / exp_sims.sum()
        
        if is_person_query:
            solo_score = float(probs[0])      # Face clearly visible, alone
            group_score = float(probs[1])     # In a group
            building_score = float(probs[2])  # No person - building/landscape
            other_person_score = float(probs[3])  # Different person
            
            # Person is visible if solo OR group score is significant
            person_visible = solo_score + group_score
            
            # Reject if it's a building/landscape without the person
            if building_score > person_visible:
                return False, solo_score, False
            
            # Reject if it's a different person entirely
            if other_person_score > person_visible:
                return False, solo_score, False
            
            # Check minimum threshold
            is_relevant = person_visible > threshold
            
            # Determine if solo
            is_solo = solo_score > group_score
            
            # In solo_only mode, reject group photos
            if solo_only and not is_solo and is_relevant:
                is_relevant = False
            
            return is_relevant, solo_score, is_solo
        else:
            # Standard relevance check
            positive_prob = float(probs[0])
            is_relevant = positive_prob > threshold
            return is_relevant, positive_prob, True
        
    except Exception as e:
        print(f"CLIP error: {e}")
        return True, 0.0, True

app = FastAPI(
    title="Google News Image Scraper API",
    description="Scrape unique, full-size images from Google News with guaranteed count",
    version="2.0.0",
)

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
    format: Optional[str] = Field(None, description="Image format")
    saved_path: Optional[str] = Field(None, description="Local path where image is saved")
    relevance_score: Optional[float] = Field(None, description="CLIP relevance score (0-1)")
    is_solo: Optional[bool] = Field(None, description="True if person is alone in image")


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
        for _ in range(10):  # Walk up to 10 levels
            if parent is None:
                break
            # Look for article, heading, or text content
            if parent.name in ["article", "div", "a", "figure"]:
                # Find headings
                heading = parent.find(["h1", "h2", "h3", "h4", "h5", "h6"])
                if heading:
                    context_text += " " + heading.get_text().lower()
                # Find figcaption
                caption = parent.find("figcaption")
                if caption:
                    context_text += " " + caption.get_text().lower()
                # Get aria-label
                aria = parent.get("aria-label", "")
                if aria:
                    context_text += " " + aria.lower()
            parent = parent.parent
        
        # Combine all context
        all_context = f"{alt_text} {title_text} {context_text}".strip()
        
        # === RELEVANCE SCORING ===
        relevance_score = 0
        if query_lower:
            # Score based on query match in context
            if query_lower in all_context:
                relevance_score += 100  # Exact phrase match
            else:
                # Check if any query word matches
                for word in query_words:
                    if len(word) > 2 and word in all_context:
                        relevance_score += 30
        
        images.append({
            "src": src,
            "estimated_width": width,
            "estimated_height": height,
            "area": width * height,
            "context": all_context[:500],  # Truncate for memory
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
            headers={"User-Agent": BROWSER_USER_AGENT},
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
    return {"status": "ok", "message": "Google News Image Scraper API v2.0 - CLIP Filtering"}


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for keep-alive pings."""
    return {"status": "healthy", "timestamp": __import__("datetime").datetime.now().isoformat()}


@app.get("/scrape", response_model=ScrapeResponse, tags=["Scraping"])
async def scrape_images(
    q: str = Query(..., min_length=1, max_length=200, description="Search term"),
    count: int = Query(default=10, ge=1, le=100, description="Number of unique images to return"),
    min_width: int = Query(default=100, ge=50, le=2000, description="Minimum image width"),
    min_height: int = Query(default=100, ge=50, le=2000, description="Minimum image height"),
    include_base64: bool = Query(default=False, description="Include base64 data"),
    formats: Optional[str] = Query(default=None, description="Allowed formats: jpg,png,webp,gif"),
    strict: bool = Query(default=True, description="Only return images that pass CLIP relevance check"),
    solo_only: bool = Query(default=True, description="For person queries, only return images of the person alone (not in groups)"),
):
    """
    Scrape exactly `count` unique images from Google News.
    
    Uses streaming approach: processes images one-by-one, stops when target reached.
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
        
        # Fetch page content with httpx (no browser needed)
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": BROWSER_USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Failed to fetch Google News: {response.status_code}")
            
            html_content = response.text
            final_url = str(response.url)
        
        # Extract all image URLs with context and relevance scoring
        image_urls = extract_image_urls(html_content, final_url, query=q)
        total_found = len(image_urls)
        
        # Stream through images until we have enough
        images: list[ImageItem] = []
        seen_hashes: set[str] = set()
        processed = 0
        
        async with httpx.AsyncClient() as client:
            for img_data in image_urls:
                if len(images) >= count:
                    break  # Early termination - we have enough!
                
                processed += 1
                
                # Strict mode: skip images without query in context
                if strict and img_data.get("relevance_score", 0) == 0:
                    continue
                
                result = await download_and_validate_image(
                    client,
                    upscale_google_news_url(img_data["src"]),  # Request full-size image
                    min_width,
                    min_height,
                    seen_hashes,
                    allowed_formats,
                )
                
                if result:
                    # CLIP semantic check - is the image actually about the query?
                    is_relevant, clip_score, is_solo = check_image_relevance(
                        result["content"], 
                        q,
                        threshold=0.2,  # 20% threshold for relevance
                        solo_only=solo_only
                    )
                    
                    if strict and not is_relevant:
                        # Skip images that don't pass CLIP check in strict mode
                        continue
                    
                    # Save to response folder
                    idx = len(images) + 1
                    filename = f"image_{idx:03d}.{result['format']}"
                    save_path = response_folder / filename
                    save_path.write_bytes(result["content"])
                    
                    # Build response item
                    item = ImageItem(
                        url=result["url"],
                        width=result["width"],
                        height=result["height"],
                        size_bytes=result["size_bytes"],
                        format=result["format"],
                        saved_path=str(save_path),
                        relevance_score=round(clip_score, 3),
                        is_solo=is_solo,
                    )
                    
                    if include_base64:
                        item.base64 = base64.b64encode(result["content"]).decode()
                    
                    images.append(item)
        
        return ScrapeResponse(
            success=True,
            query=q,
            requested_count=count,
            total_found=total_found,
            total_processed=processed,
            total_returned=len(images),
            images=images,
            response_folder=str(response_folder),
        )
        
    except Exception as e:
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
