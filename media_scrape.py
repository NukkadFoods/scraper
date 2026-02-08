"""
Media Scrape Tool - Extract and download photos/videos from webpages.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from playwright.async_api import (
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeout,
    async_playwright,
)
from playwright_stealth import Stealth

try:
    from .extractors import (
        extract_media_generic,
        extract_media_instagram,
        extract_media_twitter,
        extract_media_youtube,
    )
except ImportError:
    # If running as script directly
    from extractors import (
        extract_media_generic,
        extract_media_instagram,
        extract_media_twitter,
        extract_media_youtube,
    )

# Browser-like User-Agent
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# Default download directory
DEFAULT_DOWNLOAD_DIR = Path.cwd() / "downloads"

# Supported media extensions
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp", "svg", "bmp", "ico"}
VIDEO_EXTENSIONS = {"mp4", "webm", "ogg", "mov", "avi", "mkv", "m4v"}


def detect_platform(url: str) -> str:
    """Detect platform from URL for specialized extraction."""
    url_lower = url.lower()
    if "twitter.com" in url_lower or "x.com" in url_lower:
        return "twitter"
    elif "instagram.com" in url_lower:
        return "instagram"
    elif "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    return "generic"


def _get_domain(url: str) -> str:
    """Extract domain from URL for organizing downloads."""
    parsed = urlparse(url)
    domain = parsed.netloc or "unknown"
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _generate_filename(item: dict, index: int) -> str:
    """Generate a unique filename for a media item."""
    media_type = item.get("type", "media")
    src = item.get("src", "")
    fmt = item.get("format", "")

    # Create hash from URL for uniqueness
    url_hash = hashlib.md5(src.encode()).hexdigest()[:8]

    # Determine extension
    if fmt:
        ext = fmt.lower()
    else:
        path = urlparse(src).path
        if "." in path:
            ext = path.rsplit(".", 1)[-1].lower().split("?")[0]
        else:
            ext = "jpg" if media_type == "image" else "mp4"

    if ext not in IMAGE_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
        ext = "jpg" if media_type == "image" else "mp4"

    return f"{media_type}_{index:03d}_{url_hash}.{ext}"


GOOGLE_NEWS_LOGO_HASH = "8c30b0eaece8e454"

def _compute_dhash(image_path: str, hash_size: int = 8) -> str | None:
    """Compute difference hash for an image using Pillow (no external deps)."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
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

def _hamming_distance(h1: str, h2: str) -> int:
    """Compute Hamming distance between two hex strings."""
    if not h1 or not h2 or len(h1) != len(h2):
        return 999
    try:
        return bin(int(h1, 16) ^ int(h2, 16)).count('1')
    except ValueError:
        return 999


async def _download_media_item(
    client: httpx.AsyncClient,
    item: dict,
    output_dir: Path,
    index: int,
    min_width: int = 80,
    min_height: int = 80,
    allowed_formats: list[str] | None = None,
) -> dict:
    """Download a single media item and update with local path info."""
    src = item.get("src", "")
    if not src:
        item["error"] = "No source URL"
        return item

    try:
        response = await client.get(
            src,
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": BROWSER_USER_AGENT},
        )

        if response.status_code != 200:
            item["error"] = f"HTTP {response.status_code}"
            return item

        content = response.content
        content_type = response.headers.get("content-type", "").lower()
        
        if "text/html" in content_type or content[:15].lower().startswith((b"<!doctype", b"<html")):
            item["error"] = "Content is HTML, not image"
            return item
        
        is_png = content[:4] == b'\x89PNG'
        is_jpeg = content[:2] == b'\xff\xd8'
        is_webp = content[:4] == b'RIFF' and b'WEBP' in content[:12]
        is_gif = content[:6] in (b'GIF87a', b'GIF89a')
        
        if not (is_png or is_jpeg or is_webp or is_gif) and item.get("type") == "image":
            item["error"] = "Invalid image format"
            return item
        
        if allowed_formats:
            detected = []
            if is_png: detected.append("png")
            if is_jpeg: detected.append("jpg")
            if is_jpeg: detected.append("jpeg")
            if is_webp: detected.append("webp")
            if is_gif: detected.append("gif")
            
            if not any(fmt in allowed_formats for fmt in detected):
                item["error"] = f"Format excluded (allowed: {allowed_formats})"
                return item

        filename = _generate_filename(item, index)
        local_path = output_dir / filename

        local_path.write_bytes(content)
        
        if item.get("type") == "image":
            try:
                from PIL import Image
                import io
                with Image.open(io.BytesIO(content)) as img:
                    width, height = img.size
                    if width < min_width or height < min_height:
                        local_path.unlink(missing_ok=True)
                        item["error"] = f"Image too small: {width}x{height}"
                        return item
                    item["width"] = width
                    item["height"] = height
            except Exception:
                pass

        item["local_path"] = str(local_path)
        item["filename"] = filename
        item["size_bytes"] = len(content)

    except httpx.TimeoutException:
        item["error"] = "Download timeout"
    except httpx.RequestError as e:
        item["error"] = f"Download failed: {e!s}"
    except OSError as e:
        item["error"] = f"File write failed: {e!s}"

    return item


async def media_scrape(
    url: str,
    media_types: list[str] | None = None,
    download: bool = True,
    output_dir: str | None = None,
    max_items: int = 50,
    include_thumbnails: bool = True,
    min_width: int = 80,
    min_height: int = 80,
    allowed_formats: list[str] | None = None,
    deduplicate: bool = True,
    exclude_hashes: list[str] | None = None,
) -> dict:
    """
    Scrape and download photos/videos from a webpage.
    """
    try:
        # Validate URL
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Set defaults
        if media_types is None:
            media_types = ["image", "video"]

        # Validate and clamp max_items
        max_items = max(1, min(max_items, 500))

        # Detect platform
        platform = detect_platform(url)

        # Set up output directory
        if output_dir:
            out_path = Path(output_dir).expanduser()
        else:
            domain = _get_domain(url)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = DEFAULT_DOWNLOAD_DIR / domain / timestamp

        if download:
            out_path.mkdir(parents=True, exist_ok=True)

        # Launch headless browser with stealth
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
            try:
                context = await browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent=BROWSER_USER_AGENT,
                    locale="en-US",
                )
                page = await context.new_page()
                await Stealth().apply_stealth_async(page)

                # Navigate to page
                response = await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=60000,
                )

                # Wait for dynamic content
                await page.wait_for_timeout(3000)

                # Scroll to trigger lazy loading
                # Scroll to trigger lazy loading
                await page.evaluate("""
                    async () => {
                        await new Promise((resolve) => {
                            let totalHeight = 0;
                            const distance = 500;
                            const timer = setInterval(() => {
                                window.scrollBy(0, distance);
                                totalHeight += distance;
                                if (totalHeight >= document.body.scrollHeight || totalHeight > 5000) {
                                    clearInterval(timer);
                                    window.scrollTo(0, 0);
                                    resolve();
                                }
                            }, 100);
                        });
                    }
                """)

                await page.wait_for_timeout(1000)

                if response is None:
                    return {"error": "Navigation failed: no response received"}

                final_url = page.url
                html_content = await page.content()

            finally:
                await browser.close()

        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract media
        if platform == "twitter":
            media_items = extract_media_twitter(
                soup, final_url, media_types, include_thumbnails
            )
        elif platform == "instagram":
            media_items = extract_media_instagram(
                soup, final_url, media_types, include_thumbnails
            )
        elif platform == "youtube":
            media_items = extract_media_youtube(
                soup, final_url, media_types, include_thumbnails
            )
        else:
            media_items = extract_media_generic(
                soup, final_url, media_types, include_thumbnails
            )

        total_found = len(media_items)

        # Download media
        downloaded_count = 0
        unique_items = []
        
        blocked_hashes = set(exclude_hashes or [])
        if deduplicate:
            blocked_hashes.add(GOOGLE_NEWS_LOGO_HASH)

        if download and media_items:
            async with httpx.AsyncClient() as client:
                for i, item in enumerate(media_items[:max_items]):
                    await _download_media_item(
                        client, 
                        item, 
                        out_path, 
                        i + 1, 
                        min_width, 
                        min_height, 
                        allowed_formats
                    )
                    
                    if "local_path" in item:
                        hash_8 = _compute_dhash(item["local_path"], 8)
                        hash_4 = _compute_dhash(item["local_path"], 4)
                        
                        if hash_8 and deduplicate:
                            is_blocked = False
                            for blocked in blocked_hashes:
                                if _hamming_distance(hash_8, blocked) <= 4:
                                    is_blocked = True
                                    break
                            
                            if is_blocked:
                                Path(item["local_path"]).unlink(missing_ok=True)
                                item["error"] = "Blocked by hash"
                                del item["local_path"]
                                continue
                            
                            # Deduplicate check
                            duplicate_idx = -1
                            for idx, existing in enumerate(unique_items):
                                dist_8 = _hamming_distance(hash_8, existing["hash_8"])
                                dist_4 = _hamming_distance(hash_4, existing["hash_4"]) if hash_4 and existing["hash_4"] else 999
                                
                                if dist_8 <= 4 or dist_4 == 0:
                                    duplicate_idx = idx
                                    break
                            
                            if duplicate_idx != -1:
                                existing = unique_items[duplicate_idx]
                                new_area = item.get("width", 0) * item.get("height", 0)
                                old_area = existing["item"].get("width", 0) * existing["item"].get("height", 0)
                                
                                if new_area > old_area:
                                    Path(existing["path"]).unlink(missing_ok=True)
                                    existing["item"]["error"] = "Duplicate of larger image"
                                    if "local_path" in existing["item"]:
                                        del existing["item"]["local_path"]
                                        downloaded_count -= 1
                                    
                                    unique_items[duplicate_idx] = {
                                        "path": item["local_path"],
                                        "item": item,
                                        "hash_8": hash_8,
                                        "hash_4": hash_4
                                    }
                                    downloaded_count += 1
                                else:
                                    Path(item["local_path"]).unlink(missing_ok=True)
                                    item["error"] = "Duplicate of existing"
                                    del item["local_path"]
                            else:
                                unique_items.append({
                                    "path": item["local_path"],
                                    "item": item,
                                    "hash_8": hash_8,
                                    "hash_4": hash_4
                                })
                                downloaded_count += 1
                        else:
                            downloaded_count += 1

        result = {
            "url": url,
            "final_url": final_url,
            "platform": platform,
            "media": media_items[:max_items],
            "total_found": total_found,
            "total_downloaded": downloaded_count,
        }

        if download:
            result["output_dir"] = str(out_path)

        return result

    except PlaywrightTimeout:
        return {"error": "Page load timeout"}
    except PlaywrightError as e:
        return {"error": f"Browser error: {e!s}"}
    except Exception as e:
        return {"error": f"Scraping failed: {e!s}"}
