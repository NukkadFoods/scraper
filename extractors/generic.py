"""Generic media extractor for any webpage."""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag


def _parse_srcset(srcset: str, base_url: str) -> list[dict]:
    """Parse srcset attribute and return list of image sources with sizes."""
    sources = []
    if not srcset:
        return sources

    for item in srcset.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split()
        if parts:
            src = urljoin(base_url, parts[0])
            descriptor = parts[1] if len(parts) > 1 else ""
            sources.append({"src": src, "descriptor": descriptor})
    return sources


def _get_extension_from_url(url: str) -> str:
    """Extract file extension from URL."""
    path = urlparse(url).path
    if "." in path:
        ext = path.rsplit(".", 1)[-1].lower()
        # Clean up query params from extension
        ext = ext.split("?")[0].split("#")[0]
        if ext in ("jpg", "jpeg", "png", "gif", "webp", "svg", "bmp", "ico"):
            return ext
        if ext in ("mp4", "webm", "ogg", "mov", "avi", "mkv"):
            return ext
    return ""


def _is_valid_media_url(url: str) -> bool:
    """Check if URL is a valid media URL."""
    if not url:
        return False
    # Skip data URIs, javascript, and anchors
    if url.startswith(("data:", "javascript:", "#", "about:")):
        return False
    # Skip tiny tracking pixels and icons
    parsed = urlparse(url)
    if not parsed.scheme and not parsed.path:
        return False
    return True


def _get_best_src(element: Tag, base_url: str) -> str | None:
    """Get the best source URL from an element, handling lazy loading."""
    # Priority: data-src > data-lazy-src > data-original > src
    for attr in ("data-src", "data-lazy-src", "data-original", "src"):
        src = element.get(attr)
        if src and _is_valid_media_url(src):
            return urljoin(base_url, src)
    return None


def extract_media_generic(
    soup: BeautifulSoup,
    base_url: str,
    media_types: list[str],
    include_thumbnails: bool = True,
) -> list[dict]:
    """
    Extract media items from a generic webpage.

    Args:
        soup: BeautifulSoup parsed HTML
        base_url: Base URL for resolving relative URLs
        media_types: List of media types to extract ("image", "video")
        include_thumbnails: Whether to include video thumbnails/posters

    Returns:
        List of media item dicts with type, src, and metadata
    """
    media = []
    seen_urls = set()

    # Extract Open Graph images/videos first (usually high quality)
    if "image" in media_types:
        for meta in soup.find_all("meta", property=re.compile(r"^og:image")):
            src = meta.get("content")
            if src and _is_valid_media_url(src) and src not in seen_urls:
                seen_urls.add(src)
                media.append({
                    "type": "image",
                    "src": urljoin(base_url, src),
                    "alt": "",
                    "width": None,
                    "height": None,
                    "format": _get_extension_from_url(src) or "jpg",
                    "source": "og:image",
                })

    if "video" in media_types:
        for meta in soup.find_all("meta", property=re.compile(r"^og:video")):
            src = meta.get("content")
            if src and _is_valid_media_url(src) and src not in seen_urls:
                seen_urls.add(src)
                media.append({
                    "type": "video",
                    "src": urljoin(base_url, src),
                    "alt": "",
                    "width": None,
                    "height": None,
                    "format": _get_extension_from_url(src) or "mp4",
                    "source": "og:video",
                })

    # Extract images from <img> tags
    if "image" in media_types:
        for img in soup.find_all("img"):
            src = _get_best_src(img, base_url)
            if not src or src in seen_urls:
                continue

            seen_urls.add(src)

            # Parse dimensions
            width = img.get("width")
            height = img.get("height")
            try:
                width = int(width) if width else None
                height = int(height) if height else None
            except (ValueError, TypeError):
                width = height = None

            # Skip tiny images (likely icons/tracking pixels)
            if width and height and width < 50 and height < 50:
                continue

            media.append({
                "type": "image",
                "src": src,
                "alt": img.get("alt", ""),
                "width": width,
                "height": height,
                "format": _get_extension_from_url(src) or "jpg",
                "source": "img",
            })

            # Also check srcset for higher resolution versions
            srcset = img.get("srcset")
            if srcset:
                for srcset_item in _parse_srcset(srcset, base_url):
                    srcset_src = srcset_item["src"]
                    if srcset_src not in seen_urls:
                        seen_urls.add(srcset_src)
                        media.append({
                            "type": "image",
                            "src": srcset_src,
                            "alt": img.get("alt", ""),
                            "width": None,
                            "height": None,
                            "format": _get_extension_from_url(srcset_src) or "jpg",
                            "source": "srcset",
                            "descriptor": srcset_item.get("descriptor", ""),
                        })

    # Extract images from <picture> elements
    if "image" in media_types:
        for picture in soup.find_all("picture"):
            for source in picture.find_all("source"):
                srcset = source.get("srcset")
                if srcset:
                    for srcset_item in _parse_srcset(srcset, base_url):
                        src = srcset_item["src"]
                        if src not in seen_urls:
                            seen_urls.add(src)
                            media.append({
                                "type": "image",
                                "src": src,
                                "alt": "",
                                "width": None,
                                "height": None,
                                "format": _get_extension_from_url(src) or "jpg",
                                "source": "picture",
                                "media": source.get("media", ""),
                            })

    # Extract videos from <video> tags
    if "video" in media_types:
        for video in soup.find_all("video"):
            # Get poster image
            if include_thumbnails:
                poster = video.get("poster")
                if poster and _is_valid_media_url(poster) and poster not in seen_urls:
                    seen_urls.add(poster)
                    media.append({
                        "type": "image",
                        "src": urljoin(base_url, poster),
                        "alt": "Video poster",
                        "width": video.get("width"),
                        "height": video.get("height"),
                        "format": _get_extension_from_url(poster) or "jpg",
                        "source": "video-poster",
                    })

            # Get video source
            src = _get_best_src(video, base_url)
            if src and src not in seen_urls:
                seen_urls.add(src)

                width = video.get("width")
                height = video.get("height")
                try:
                    width = int(width) if width else None
                    height = int(height) if height else None
                except (ValueError, TypeError):
                    width = height = None

                media.append({
                    "type": "video",
                    "src": src,
                    "alt": "",
                    "width": width,
                    "height": height,
                    "format": _get_extension_from_url(src) or "mp4",
                    "source": "video",
                })

            # Check <source> children
            for source in video.find_all("source"):
                src = source.get("src")
                if src and _is_valid_media_url(src) and src not in seen_urls:
                    seen_urls.add(src)
                    media.append({
                        "type": "video",
                        "src": urljoin(base_url, src),
                        "alt": "",
                        "width": None,
                        "height": None,
                        "format": _get_extension_from_url(src) or source.get("type", "").split("/")[-1] or "mp4",
                        "source": "video-source",
                    })

    # Extract background images from inline styles (common pattern)
    if "image" in media_types:
        bg_pattern = re.compile(r'url\(["\']?([^"\')\s]+)["\']?\)')
        for element in soup.find_all(style=True):
            style = element.get("style", "")
            for match in bg_pattern.findall(style):
                if _is_valid_media_url(match) and match not in seen_urls:
                    src = urljoin(base_url, match)
                    seen_urls.add(src)
                    media.append({
                        "type": "image",
                        "src": src,
                        "alt": "",
                        "width": None,
                        "height": None,
                        "format": _get_extension_from_url(src) or "jpg",
                        "source": "background-image",
                    })

    return media
