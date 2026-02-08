"""Instagram media extractor."""

from __future__ import annotations

import json
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup


def extract_media_instagram(
    soup: BeautifulSoup,
    base_url: str,
    media_types: list[str],
    include_thumbnails: bool = True,
) -> list[dict]:
    """
    Extract media items from an Instagram page.

    Handles:
    - Post images and videos
    - Profile pictures
    - Story thumbnails
    - Reels thumbnails

    Note: Instagram heavily uses JavaScript, so data is often in embedded JSON.

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

    # Try to extract from embedded JSON/script data
    for script in soup.find_all("script", type="application/json"):
        try:
            data = json.loads(script.string or "")
            media.extend(_extract_from_json(data, media_types, seen_urls, include_thumbnails))
        except (json.JSONDecodeError, TypeError):
            continue

    # Also try scripts with specific patterns
    for script in soup.find_all("script"):
        if script.string:
            # Look for window._sharedData or similar
            match = re.search(r"window\._sharedData\s*=\s*(\{.+?\});", script.string)
            if match:
                try:
                    data = json.loads(match.group(1))
                    media.extend(_extract_from_json(data, media_types, seen_urls, include_thumbnails))
                except (json.JSONDecodeError, TypeError):
                    pass

            # Look for __additionalDataLoaded
            match = re.search(r"__additionalDataLoaded\s*\(\s*['\"].*?['\"]\s*,\s*(\{.+?\})\s*\)", script.string)
            if match:
                try:
                    data = json.loads(match.group(1))
                    media.extend(_extract_from_json(data, media_types, seen_urls, include_thumbnails))
                except (json.JSONDecodeError, TypeError):
                    pass

    # Extract Open Graph images (fallback)
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

    if "video" in media_types:
        for meta in soup.find_all("meta", property="og:video"):
            src = meta.get("content")
            if src and src not in seen_urls:
                seen_urls.add(src)
                media.append({
                    "type": "video",
                    "src": src,
                    "alt": "",
                    "width": None,
                    "height": None,
                    "format": "mp4",
                    "source": "og:video",
                })

    # Direct image extraction (profile pics, etc.)
    if "image" in media_types:
        for img in soup.find_all("img"):
            src = img.get("src")
            if src and "cdninstagram.com" in src and src not in seen_urls:
                seen_urls.add(src)
                media.append({
                    "type": "image",
                    "src": src,
                    "alt": img.get("alt", ""),
                    "width": None,
                    "height": None,
                    "format": "jpg",
                    "source": "img",
                })

    # Direct video extraction
    if "video" in media_types:
        for video in soup.find_all("video"):
            src = video.get("src")
            if src and src not in seen_urls:
                seen_urls.add(src)
                media.append({
                    "type": "video",
                    "src": src,
                    "alt": "",
                    "width": None,
                    "height": None,
                    "format": "mp4",
                    "source": "video",
                })

            # Get poster
            if include_thumbnails:
                poster = video.get("poster")
                if poster and poster not in seen_urls:
                    seen_urls.add(poster)
                    media.append({
                        "type": "image",
                        "src": poster,
                        "alt": "Video thumbnail",
                        "width": None,
                        "height": None,
                        "format": "jpg",
                        "source": "video-poster",
                    })

    return media


def _extract_from_json(
    data: dict | list,
    media_types: list[str],
    seen_urls: set,
    include_thumbnails: bool,
) -> list[dict]:
    """Recursively extract media from Instagram's JSON data."""
    media = []

    if isinstance(data, dict):
        # Check for display_url (images)
        if "image" in media_types:
            for key in ("display_url", "display_src", "thumbnail_src", "profile_pic_url"):
                url = data.get(key)
                if url and isinstance(url, str) and url not in seen_urls:
                    seen_urls.add(url)
                    media.append({
                        "type": "image",
                        "src": url,
                        "alt": data.get("accessibility_caption", ""),
                        "width": data.get("dimensions", {}).get("width") if isinstance(data.get("dimensions"), dict) else None,
                        "height": data.get("dimensions", {}).get("height") if isinstance(data.get("dimensions"), dict) else None,
                        "format": "jpg",
                        "source": f"json-{key}",
                    })

            # Check display_resources for multiple resolutions
            if "display_resources" in data and isinstance(data["display_resources"], list):
                for resource in data["display_resources"]:
                    if isinstance(resource, dict):
                        url = resource.get("src")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            media.append({
                                "type": "image",
                                "src": url,
                                "alt": "",
                                "width": resource.get("config_width"),
                                "height": resource.get("config_height"),
                                "format": "jpg",
                                "source": "json-display_resources",
                            })

        # Check for video_url
        if "video" in media_types:
            video_url = data.get("video_url")
            if video_url and isinstance(video_url, str) and video_url not in seen_urls:
                seen_urls.add(video_url)
                media.append({
                    "type": "video",
                    "src": video_url,
                    "alt": "",
                    "width": data.get("dimensions", {}).get("width") if isinstance(data.get("dimensions"), dict) else None,
                    "height": data.get("dimensions", {}).get("height") if isinstance(data.get("dimensions"), dict) else None,
                    "format": "mp4",
                    "source": "json-video_url",
                })

        # Check for carousel (sidecar) items
        if "edge_sidecar_to_children" in data:
            edges = data["edge_sidecar_to_children"].get("edges", [])
            for edge in edges:
                if isinstance(edge, dict) and "node" in edge:
                    media.extend(_extract_from_json(edge["node"], media_types, seen_urls, include_thumbnails))

        # Recurse into nested structures
        for value in data.values():
            if isinstance(value, (dict, list)):
                media.extend(_extract_from_json(value, media_types, seen_urls, include_thumbnails))

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                media.extend(_extract_from_json(item, media_types, seen_urls, include_thumbnails))

    return media
