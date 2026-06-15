"""Normalize community report titles: remove generic prefixes models echo from prompts."""

from __future__ import annotations

import re

# Headlines copied from GraphRAG-style prompts (e.g. "dinamica comunității") make every
# community sound identical; strip them when followed by substantive text.
_GENERIC_TITLE_OPENING = re.compile(
    r"(?is)^\s*(?:"
    r"dinamica\s+comunit(?:ă|a)ții|"
    r"dinamica\s+comunitatii|"
    r"dinamica\s+grupului(?:\s+social)?|"
    r"perspectiva\s+comunit(?:ă|a)ții|"
    r"perspectiva\s+comunitatii|"
    r"analiza\s+comunit(?:ă|a)ții|"
    r"analiza\s+comunitatii|"
    r"analiza\s+nara(?:ț|t)iunii|"
    r"descrierea\s+comunit(?:ă|a)ții|"
    r"descrierea\s+comunitatii|"
    r"rolul\s+comunit(?:ă|a)ții(?:\s+în)?|"
    r"rolul\s+comunitatii(?:\s+în)?|"
    r"tema\s+principal(?:ă|a)\s+a\s+comunit(?:ă|a)ții|"
    r"nucleul\s+nara(?:ț|t)iv(?:\s+al\s+comunit(?:ă|a)ții)?|"
    r"comunitatea\s+despre"
    r")(\s*[:\-–—]\s*|\s+)"
)


def strip_generic_community_title_prefix(title: str, *, min_rest_len: int = 6) -> str:
    """Remove leading boilerplate from a title; keep original if the remainder is too short."""
    if not title or not isinstance(title, str):
        return title if isinstance(title, str) else ""
    t = " ".join(title.split())
    m = _GENERIC_TITLE_OPENING.match(t)
    if not m:
        return t
    rest = t[m.end() :].strip()
    if len(rest) >= min_rest_len:
        return rest
    return t
