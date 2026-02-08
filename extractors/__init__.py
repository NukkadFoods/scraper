"""Media extractors for different platforms."""

from .generic import extract_media_generic
from .twitter import extract_media_twitter
from .instagram import extract_media_instagram
from .youtube import extract_media_youtube

__all__ = [
    "extract_media_generic",
    "extract_media_twitter",
    "extract_media_instagram",
    "extract_media_youtube",
]
