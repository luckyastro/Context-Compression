from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from cc_tokenizers import Tokenizer, count_tokens, truncate_to_token_budget


_RE_SENTENCE = re.compile(r"(?<=[\.\?\!])\s+|\n{2,}")
_RE_WORD = re.compile(r"[A-Za-z]{2,}")

_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "you",
    "your",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "not",
    "but",
    "they",
    "their",
    "its",
    "into",
    "about",
    "than",
    "then",
    "them",
    "also",
    "can",
    "will",
    "would",
    "could",
    "should",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "how",
    "a",
    "an",
    "to",
    "of",
    "in",
    "on",
    "as",
    "is",
    "it",
    "at",
    "be",
    "by",
    "or",
    "if",
}


def split_sentences(text: str) -> List[str]:
    chunks = [c.strip() for c in _RE_SENTENCE.split(text) if c.strip()]
    out: List[str] = []
    for c in chunks:
        if len(c) > 1200 and "\n" in c:
            out.extend([x.strip() for x in c.splitlines() if x.strip()])
        else:
            out.append(c)
    return out


def _word_counts(sentences: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in sentences:
        for w in _RE_WORD.findall(s.lower()):
            if w in _STOPWORDS:
                continue
            counts[w] = counts.get(w, 0) + 1
    return counts


def _sentence_score(s: str, counts: Dict[str, int]) -> float:
    words = [w for w in _RE_WORD.findall(s.lower()) if w not in _STOPWORDS]
    if not words:
        return 0.0
    return sum(math.log1p(counts.get(w, 0)) for w in words) / math.sqrt(len(words))


@dataclass(frozen=True)
class ExtractiveResult:
    text: str
    target_tokens: int
    output_tokens: int
    selected_sentences: int
    total_sentences: int


def extractive_compress(
    text: str,
    *,
    tokenizer: Tokenizer,
    target_tokens: int,
    min_sentence_chars: int = 25,
) -> ExtractiveResult:
    if target_tokens <= 0 or not text.strip():
        return ExtractiveResult("", target_tokens, 0, 0, 0)

    sentences = split_sentences(text)
    total = len(sentences)
    if total == 0:
        truncated = truncate_to_token_budget(text, tokenizer, target_tokens)
        return ExtractiveResult(truncated, target_tokens, count_tokens(truncated, tokenizer), 1, 1)

    counts = _word_counts(sentences)
    scored: List[Tuple[int, float]] = []
    for i, s in enumerate(sentences):
        if len(s) < min_sentence_chars:
            continue
        scored.append((i, _sentence_score(s, counts)))

    if not scored:
        scored = [(i, 1.0) for i in range(total)]

    scored.sort(key=lambda x: x[1], reverse=True)

    selected_idx: List[int] = []
    current_tokens = 0
    for i, _ in scored:
        s_tok = count_tokens(sentences[i], tokenizer)
        if selected_idx and current_tokens + s_tok > target_tokens:
            continue
        selected_idx.append(i)
        current_tokens += s_tok
        if current_tokens >= target_tokens:
            break

    selected_set = set(selected_idx)
    if current_tokens < target_tokens:
        for i, _ in scored:
            if i in selected_set:
                continue
            s_tok = count_tokens(sentences[i], tokenizer)
            if current_tokens + s_tok > target_tokens:
                continue
            selected_set.add(i)
            current_tokens += s_tok
            if current_tokens >= target_tokens:
                break

    selected_idx = sorted(selected_set)
    out = "\n\n".join(sentences[i] for i in selected_idx).strip()
    out = truncate_to_token_budget(out, tokenizer, target_tokens)
    out_tokens = count_tokens(out, tokenizer)
    return ExtractiveResult(out, target_tokens, out_tokens, len(selected_idx), total)

