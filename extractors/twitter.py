"""Twitter/X media extractor."""

from __future__ import annotations

import json
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup


def extract_media_twitter(
    soup: BeautifulSoup,
    base_url: str,
    media_types: list[str],
    include_thumbnails: bool = True,
) -> list[dict]:
    """
    Extract media items from a Twitter/X page.

    Handles:
    - Tweet images (data-testid="tweetPhoto")
    - Tweet videos
    - Profile images
    - Embedded media in tweets

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

    # Try to extract from embedded JSON data (more reliable)
    for script in soup.find_all("script", type="application/json"):
        try:
            data = json.loads(script.string or "")
            media.extend(_extract_from_json(data, media_types, seen_urls))
        except (json.JSONDecodeError, TypeError):
            continue

    # Extract tweet photos using data-testid
    if "image" in media_types:
        # Tweet images
        for img_container in soup.find_all(attrs={"data-testid": "tweetPhoto"}):
            for img in img_container.find_all("img"):
                src = img.get("src")
                if src and src not in seen_urls:
                    # Twitter image URLs - get highest quality
                    src = _get_best_quality_twitter_image(src)
                    seen_urls.add(src)
                    media.append({
                        "type": "image",
                        "src": src,
                        "alt": img.get("alt", ""),
                        "width": None,
                        "height": None,
                        "format": "jpg",
                        "source": "tweet-photo",
                    })

        # Profile images
        for img in soup.find_all("img", src=re.compile(r"pbs\.twimg\.com/profile_images")):
            src = img.get("src")
            if src and src not in seen_urls:
                src = _get_best_quality_twitter_image(src)
                seen_urls.add(src)
                media.append({
                    "type": "image",
                    "src": src,
                    "alt": img.get("alt", "Profile image"),
                    "width": None,
                    "height": None,
                    "format": "jpg",
                    "source": "profile-image",
                })

        # Card images
        for img in soup.find_all("img", src=re.compile(r"pbs\.twimg\.com/card_img")):
            src = img.get("src")
            if src and src not in seen_urls:
                seen_urls.add(src)
                media.append({
                    "type": "image",
                    "src": src,
                    "alt": img.get("alt", ""),
                    "width": None,
                    "height": None,
                    "format": "jpg",
                    "source": "card-image",
                })

    # Extract videos
    if "video" in media_types:
        for video in soup.find_all("video"):
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

            # Get video source
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
                    "source": "tweet-video",
                })

            # Check source tags
            for source in video.find_all("source"):
                src = source.get("src")
                if src and src not in seen_urls:
                    seen_urls.add(src)
                    media.append({
                        "type": "video",
                        "src": src,
                        "alt": "",
                        "width": None,
                        "height": None,
                        "format": source.get("type", "video/mp4").split("/")[-1],
                        "source": "tweet-video",
                    })

    return media


def _get_best_quality_twitter_image(url: str) -> str:
    """Convert Twitter image URL to highest quality version."""
    # Twitter uses format like: pbs.twimg.com/media/xxx?format=jpg&name=small
    # We want: pbs.twimg.com/media/xxx?format=jpg&name=large or name=orig
    if "twimg.com" in url:
        # Remove size suffix and get original
        url = re.sub(r"&name=\w+", "&name=orig", url)
        url = re.sub(r"\?name=\w+", "?name=orig", url)
        # If no name param, add it
        if "name=" not in url:
            if "?" in url:
                url += "&name=orig"
            else:
                url += "?name=orig"
    return url


def _extract_from_json(data: dict | list, media_types: list[str], seen_urls: set) -> list[dict]:
    """Recursively extract media from Twitter's JSON data."""
    media = []

    if isinstance(data, dict):
        # Check for media entities
        if "media" in data and isinstance(data["media"], list):
            for item in data["media"]:
                if isinstance(item, dict):
                    media_type = item.get("type", "")
                    if media_type == "photo" and "image" in media_types:
                        url = item.get("media_url_https") or item.get("media_url")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            media.append({
                                "type": "image",
                                "src": url + "?name=orig",
                                "alt": item.get("ext_alt_text", ""),
                                "width": item.get("original_info", {}).get("width"),
                                "height": item.get("original_info", {}).get("height"),
                                "format": "jpg",
                                "source": "json-media",
                            })
                    elif media_type in ("video", "animated_gif") and "video" in media_types:
                        variants = item.get("video_info", {}).get("variants", [])
                        # Get highest bitrate variant
                        best_variant = None
                        best_bitrate = -1
                        for variant in variants:
                            if variant.get("content_type") == "video/mp4":
                                bitrate = variant.get("bitrate", 0)
                                if bitrate > best_bitrate:
                                    best_bitrate = bitrate
                                    best_variant = variant
                        if best_variant:
                            url = best_variant.get("url")
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                media.append({
                                    "type": "video",
                                    "src": url,
                                    "alt": "",
                                    "width": item.get("original_info", {}).get("width"),
                                    "height": item.get("original_info", {}).get("height"),
                                    "format": "mp4",
                                    "source": "json-video",
                                })

        # Recurse into nested dicts
        for value in data.values():
            media.extend(_extract_from_json(value, media_types, seen_urls))

    elif isinstance(data, list):
        for item in data:
            media.extend(_extract_from_json(item, media_types, seen_urls))

    return media
