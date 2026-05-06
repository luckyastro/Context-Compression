from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import List, Tuple


_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_MULTI_WS = re.compile(r"[ \t]+")
_RE_MANY_NEWLINES = re.compile(r"\n{3,}")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")


def _non_alnum_ratio(s: str) -> float:
    if not s:
        return 0.0
    non_alnum = sum(1 for c in s if not c.isalnum() and not c.isspace())
    return non_alnum / max(1, len(s))


def _looks_like_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True

    # dataset artifacts embedded inside `challenge_text`
    if s.startswith(("Passage:", "Question:", "Answer:")):
        return True

    lower = s.lower()
    boiler = (
        "cookie",
        "privacy policy",
        "terms of service",
        "all rights reserved",
        "subscribe",
        "sign in",
        "log in",
        "newsletter",
        "accept all",
        "reject all",
        "advertisement",
        "share this",
    )
    if any(b in lower for b in boiler):
        return True

    if _RE_URL.search(s) and len(s) < 160:
        return True

    if len(s) >= 20 and _non_alnum_ratio(s) > 0.55:
        return True

    words = re.findall(r"[A-Za-z]{3,}", s)
    if len(s) < 8 and not words:
        return True

    return False


@dataclass(frozen=True)
class DenoiseStats:
    input_lines: int
    kept_lines: int
    removed_lines: int
    deduped_lines: int


def denoise_text(text: str, *, aggressive: bool = False) -> Tuple[str, DenoiseStats]:
    if not text:
        return "", DenoiseStats(0, 0, 0, 0)

    t = html.unescape(text)
    t = _RE_HTML_TAG.sub(" ", t)

    lines_in = t.splitlines()
    seen = set()
    kept: List[str] = []
    removed = 0
    deduped = 0

    for line in lines_in:
        raw = line.strip()
        if not raw:
            continue

        norm = _RE_MULTI_WS.sub(" ", raw)
        key = norm.lower()

        if key in seen:
            deduped += 1
            continue
        seen.add(key)

        if _looks_like_noise_line(norm):
            removed += 1
            continue

        if aggressive:
            if len(norm) >= 12 and len(re.findall(r"[A-Za-z]", norm)) == 0:
                removed += 1
                continue
            if len(norm) <= 32 and sum(1 for c in norm if c.isupper()) >= 8:
                removed += 1
                continue

        kept.append(norm)

    out = "\n".join(kept)
    out = _RE_MULTI_WS.sub(" ", out)
    out = _RE_MANY_NEWLINES.sub("\n\n", out).strip()

    return out, DenoiseStats(
        input_lines=len(lines_in),
        kept_lines=len(kept),
        removed_lines=removed,
        deduped_lines=deduped,
    )

