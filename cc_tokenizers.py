from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Protocol


class Tokenizer(Protocol):
    def encode(self, text: str) -> List[int]: ...

    def decode(self, tokens: List[int]) -> str: ...


_WORDLIKE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class SimpleRegexTokenizer:
    """
    Fallback tokenizer when no external tokenizer libs are available.

    Notes:
    - This is NOT model-equivalent tokenization; it is only used to enforce
      approximate token-ratio budgets consistently.
    - `encode()` returns stable integer ids for each distinct surface token.
    """

    def encode(self, text: str) -> List[int]:
        parts = _WORDLIKE.findall(text)
        vocab: dict[str, int] = {}
        out: List[int] = []
        for p in parts:
            if p not in vocab:
                vocab[p] = len(vocab) + 1
            out.append(vocab[p])
        return out

    def decode(self, tokens: List[int]) -> str:
        raise NotImplementedError("SimpleRegexTokenizer cannot decode tokens.")


@dataclass(frozen=True)
class TiktokenTokenizer:
    encoding_name: str = "o200k_base"

    def __post_init__(self) -> None:
        import tiktoken  # type: ignore

        object.__setattr__(self, "_enc", tiktoken.get_encoding(self.encoding_name))

    def encode(self, text: str) -> List[int]:
        return list(self._enc.encode(text))

    def decode(self, tokens: List[int]) -> str:
        return self._enc.decode(tokens)


@dataclass(frozen=True)
class HFTokenizer:
    model_name_or_path: str = "gpt2"

    def __post_init__(self) -> None:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        object.__setattr__(self, "_tok", tok)

    def encode(self, text: str) -> List[int]:
        return list(self._tok.encode(text, add_special_tokens=False))

    def decode(self, tokens: List[int]) -> str:
        return self._tok.decode(tokens)


def get_tokenizer(
    preferred: Optional[str] = None,
    hf_model_name_or_path: str = "gpt2",
    tiktoken_encoding: str = "o200k_base",
) -> Tokenizer:
    """
    Returns the best available tokenizer.

    preferred:
      - "tiktoken" | "hf" | "simple" | None (auto)
    """
    preferred = (preferred or "").strip().lower() or None

    if preferred in (None, "tiktoken"):
        try:
            return TiktokenTokenizer(encoding_name=tiktoken_encoding)
        except Exception:
            if preferred == "tiktoken":
                raise

    if preferred in (None, "hf", "transformers"):
        try:
            return HFTokenizer(model_name_or_path=hf_model_name_or_path)
        except Exception:
            if preferred in ("hf", "transformers"):
                raise

    return SimpleRegexTokenizer()


def count_tokens(text: str, tokenizer: Tokenizer) -> int:
    return len(tokenizer.encode(text))


def truncate_to_token_budget(text: str, tokenizer: Tokenizer, budget_tokens: int) -> str:
    if budget_tokens <= 0:
        return ""
    try:
        toks = tokenizer.encode(text)
        if len(toks) <= budget_tokens:
            return text
        return tokenizer.decode(toks[:budget_tokens])
    except Exception:
        parts = _WORDLIKE.findall(text)
        if len(parts) <= budget_tokens:
            return text
        out: List[str] = []
        for p in parts[:budget_tokens]:
            if not out:
                out.append(p)
            elif re.match(r"[^\w\s]", p) and p not in ("(", "[", "{"):
                out[-1] = out[-1] + p
            else:
                out.append(" " + p)
        return "".join(out)

