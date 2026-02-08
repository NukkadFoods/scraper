"""YouTube media extractor."""

from __future__ import annotations

import json
import re
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup


# YouTube thumbnail URL templates
YOUTUBE_THUMB_QUALITIES = [
    ("maxresdefault", 1280, 720),
    ("sddefault", 640, 480),
    ("hqdefault", 480, 360),
    ("mqdefault", 320, 180),
    ("default", 120, 90),
]


def extract_media_youtube(
    soup: BeautifulSoup,
    base_url: str,
    media_types: list[str],
    include_thumbnails: bool = True,
) -> list[dict]:
    """
    Extract media items from a YouTube page.

    Handles:
    - Video thumbnails (multiple resolutions)
    - Channel art/banners
    - Profile pictures

    Note: Direct video download URLs are not available via HTML scraping.
    This extractor focuses on thumbnails and images.

    Args:
        soup: BeautifulSoup parsed HTML
        base_url: Base URL for resolving relative URLs
        media_types: List of media types to extract ("image", "video")
        include_thumbnails: Whether to include video thumbnails

    Returns:
        List of media item dicts
    """
    media = []
    seen_urls = set()

    # Extract video ID from URL
    video_id = _extract_video_id(base_url)

    # Generate thumbnail URLs for main video
    if video_id and "image" in media_types and include_thumbnails:
        for quality, width, height in YOUTUBE_THUMB_QUALITIES:
            thumb_url = f"https://i.ytimg.com/vi/{video_id}/{quality}.jpg"
            if thumb_url not in seen_urls:
                seen_urls.add(thumb_url)
                media.append({
                    "type": "image",
                    "src": thumb_url,
                    "alt": f"Video thumbnail ({quality})",
                    "width": width,
                    "height": height,
                    "format": "jpg",
                    "source": f"thumbnail-{quality}",
                    "video_id": video_id,
                })

        # WebP thumbnails (usually available)
        for quality, width, height in YOUTUBE_THUMB_QUALITIES[:3]:  # Top 3 qualities
            thumb_url = f"https://i.ytimg.com/vi_webp/{video_id}/{quality}.webp"
            if thumb_url not in seen_urls:
                seen_urls.add(thumb_url)
                media.append({
                    "type": "image",
                    "src": thumb_url,
                    "alt": f"Video thumbnail ({quality}, WebP)",
                    "width": width,
                    "height": height,
                    "format": "webp",
                    "source": f"thumbnail-{quality}-webp",
                    "video_id": video_id,
                })

    # Extract from JSON embedded data
    for script in soup.find_all("script"):
        if script.string:
            # Look for ytInitialData or ytInitialPlayerResponse
            for pattern in [
                r"var\s+ytInitialData\s*=\s*(\{.+?\});",
                r"ytInitialData\s*=\s*(\{.+?\});",
                r"var\s+ytInitialPlayerResponse\s*=\s*(\{.+?\});",
            ]:
                match = re.search(pattern, script.string)
                if match:
                    try:
                        data = json.loads(match.group(1))
                        media.extend(_extract_from_json(data, media_types, seen_urls, include_thumbnails))
                    except (json.JSONDecodeError, TypeError):
                        pass

    # Extract Open Graph images
    if "image" in media_types:
        for meta in soup.find_all("meta", property="og:image"):
            src = meta.get("content")
            if src and src not in seen_urls:
                seen_urls.add(src)
                media.append({
                    "type": "image",
                    "src": src,
                    "alt": "",
                    "width": None,
                    "height": None,
                    "format": "jpg",
                    "source": "og:image",
                })

    # Extract channel art and profile pics from page
    if "image" in media_types:
        # Channel banners
        for img in soup.find_all("img", src=re.compile(r"yt3\.ggpht\.com")):
            src = img.get("src")
            if src and src not in seen_urls:
                seen_urls.add(src)
                media.append({
                    "type": "image",
                    "src": src,
                    "alt": img.get("alt", "Channel image"),
                    "width": None,
                    "height": None,
                    "format": "jpg",
                    "source": "channel-image",
                })

        # Video thumbnails in listings
        for img in soup.find_all("img", src=re.compile(r"i\.ytimg\.com")):
            src = img.get("src")
            if src and src not in seen_urls:
                seen_urls.add(src)
                # Try to extract video ID from thumbnail URL
                vid_match = re.search(r"/vi(?:_webp)?/([^/]+)/", src)
                media.append({
                    "type": "image",
                    "src": src,
                    "alt": img.get("alt", "Video thumbnail"),
                    "width": None,
                    "height": None,
                    "format": "webp" if "webp" in src else "jpg",
                    "source": "video-thumbnail",
                    "video_id": vid_match.group(1) if vid_match else None,
                })

    return media


def _extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    parsed = urlparse(url)

    # youtube.com/watch?v=VIDEO_ID
    if "youtube.com" in parsed.netloc:
        query = parse_qs(parsed.query)
        if "v" in query:
            return query["v"][0]
        # youtube.com/embed/VIDEO_ID
        if "/embed/" in parsed.path:
            parts = parsed.path.split("/embed/")
            if len(parts) > 1:
                return parts[1].split("/")[0].split("?")[0]
        # youtube.com/v/VIDEO_ID
        if "/v/" in parsed.path:
            parts = parsed.path.split("/v/")
            if len(parts) > 1:
                return parts[1].split("/")[0].split("?")[0]

    # youtu.be/VIDEO_ID
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/").split("?")[0]

    return None


def _extract_from_json(
    data: dict | list,
    media_types: list[str],
    seen_urls: set,
    include_thumbnails: bool,
) -> list[dict]:
    """Recursively extract media from YouTube's JSON data."""
    media = []

    if isinstance(data, dict):
        # Check for thumbnail objects
        if "image" in media_types and include_thumbnails:
            if "thumbnails" in data and isinstance(data["thumbnails"], list):
                for thumb in data["thumbnails"]:
                    if isinstance(thumb, dict):
                        url = thumb.get("url")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            media.append({
                                "type": "image",
                                "src": url,
                                "alt": "Thumbnail",
                                "width": thumb.get("width"),
                                "height": thumb.get("height"),
                                "format": "webp" if "webp" in url else "jpg",
                                "source": "json-thumbnail",
                            })

            # Check for avatar/banner images
            for key in ("avatar", "banner", "channelBanner"):
                if key in data and isinstance(data[key], dict):
                    thumbs = data[key].get("thumbnails", [])
                    for thumb in thumbs:
                        if isinstance(thumb, dict):
                            url = thumb.get("url")
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                media.append({
                                    "type": "image",
                                    "src": url,
                                    "alt": f"Channel {key}",
                                    "width": thumb.get("width"),
                                    "height": thumb.get("height"),
                                    "format": "jpg",
                                    "source": f"json-{key}",
                                })

        # Recurse into nested structures
        for value in data.values():
            if isinstance(value, (dict, list)):
                media.extend(_extract_from_json(value, media_types, seen_urls, include_thumbnails))

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                media.extend(_extract_from_json(item, media_types, seen_urls, include_thumbnails))

    return media
