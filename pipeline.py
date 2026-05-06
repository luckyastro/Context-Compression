from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from denoise import DenoiseStats, denoise_text
from extractive import ExtractiveResult, extractive_compress
from cc_tokenizers import Tokenizer, count_tokens, get_tokenizer
from local_model_stage import LocalModelConfig, compress_with_local_hf_model


RatioLike = Union[float, int, str]


def parse_ratios(values: Sequence[RatioLike]) -> List[float]:
    """
    Normalize ratio inputs into floats in (0, 1).

    Accepts:
    - floats like 0.2
    - ints like 20 (interpreted as 20%)
    - strings like "0.2", "20%", "20"
    """
    out: List[float] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, (float, int)):
            x = float(v)
        else:
            s = str(v).strip()
            if not s:
                continue
            s = s.replace("%", "")
            x = float(s)
        if x > 1.0:
            x = x / 100.0
        out.append(x)
    return out


@dataclass(frozen=True)
class CompressionRun:
    ratio: float
    input_tokens: int
    target_tokens: int
    output_tokens: int
    denoise_stats: DenoiseStats
    extractive: ExtractiveResult
    stage: str  # extractive | local_hf | icae
    output_text: str


def compress_to_ratios(
    text: str,
    *,
    ratios: Sequence[RatioLike] = (0.2, 0.4, 0.6),
    tokenizer_preferred: Optional[str] = None,
    hf_model_name_or_path: str = "gpt2",
    tiktoken_encoding: str = "o200k_base",
    denoise: bool = True,
    denoise_aggressive: bool = False,
    min_target_tokens: int = 1,
    # optional local rewrite stage (CUDA-capable via torch/transformers)
    use_local_hf: bool = False,
    local_hf_cfg: Optional[LocalModelConfig] = None,
    # optional ICAE stage (CUDA-capable; requires deps + checkpoint)
    use_icae: bool = False,
    icae_base_model_name_or_path: str = "",
    icae_checkpoint_safetensors_path: str = "",
    icae_device: str = "auto",
    icae_use_for_ratios_leq: float = 0.25,
    icae_load_in_8bit: bool = False,
    icae_load_in_4bit: bool = False,
    icae_device_map: str = "none",
    icae_offload_folder: str = "offload",
    icae_max_input_tokens: int = 5120,
    icae_max_generate_tokens: int = 512,
) -> Tuple[List[CompressionRun], Tokenizer]:
    tok = get_tokenizer(
        preferred=tokenizer_preferred,
        hf_model_name_or_path=hf_model_name_or_path,
        tiktoken_encoding=tiktoken_encoding,
    )

    cleaned = text
    denoise_stats = DenoiseStats(0, 0, 0, 0)
    if denoise:
        cleaned, denoise_stats = denoise_text(text, aggressive=denoise_aggressive)

    input_tokens = count_tokens(cleaned, tok)

    ratios_f = parse_ratios(ratios) or [0.2, 0.4, 0.6]

    # ICAE is heavy: initialize once (lazy) and reuse.
    icae_compressor = None

    runs: List[CompressionRun] = []
    for r in ratios_f:
        if r <= 0 or r >= 1.0:
            raise ValueError(f"ratio must be between 0 and 1, got {r}")
        target = max(min_target_tokens, int(round(input_tokens * r)))
        ext = extractive_compress(cleaned, tokenizer=tok, target_tokens=target)

        out_text = ext.text
        stage = "extractive"

        # Optional local HF rewrite stage (helps retain QA-relevant facts while compressing).
        if use_local_hf:
            cfg = local_hf_cfg or LocalModelConfig()
            out_text = compress_with_local_hf_model(
                out_text,
                target_tokens=target,
                cfg=cfg,
                budget_tokenizer_name_or_path=hf_model_name_or_path if tokenizer_preferred == "hf" else None,
            )
            stage = "local_hf"

        # Optional ICAE stage for very high compression (<= 25% by default).
        if use_icae and r <= icae_use_for_ratios_leq:
            if not icae_base_model_name_or_path or not icae_checkpoint_safetensors_path:
                raise ValueError("ICAE enabled but base model or checkpoint path not provided.")
            from icae_min import ICAECompressor, ICAEConfig
            try:
                if icae_compressor is None:
                    icae_compressor = ICAECompressor(
                        ICAEConfig(
                            base_model_name_or_path=icae_base_model_name_or_path,
                            checkpoint_safetensors_path=icae_checkpoint_safetensors_path,
                            device=icae_device,
                            max_input_tokens=icae_max_input_tokens,
                            max_generate_tokens=icae_max_generate_tokens,
                            load_in_8bit=icae_load_in_8bit,
                            load_in_4bit=icae_load_in_4bit,
                            device_map=icae_device_map,
                            offload_folder=icae_offload_folder,
                        )
                    )
            except RuntimeError as e:
                # Re-raise with context about why ICAE was invoked.
                raise RuntimeError(
                    f"ICAE stage failed to initialize (ratio={r}, target_tokens={target}).\n\n{e}"
                ) from e
            out_text = icae_compressor.compress_generate(out_text)
            stage = "icae"

            # Hard trim again using our counting tokenizer to stay on-ratio.
            # (ICAE generation length is controlled by max_generate_tokens but not exact.)
            from cc_tokenizers import truncate_to_token_budget

            out_text = truncate_to_token_budget(out_text, tok, target).strip()

        out_tokens = count_tokens(out_text, tok)
        runs.append(
            CompressionRun(
                ratio=float(r),
                input_tokens=input_tokens,
                target_tokens=ext.target_tokens,
                output_tokens=out_tokens,
                denoise_stats=denoise_stats,
                extractive=ext,
                stage=stage,
                output_text=out_text,
            )
        )

    return runs, tok


def write_runs(
    runs: Sequence[CompressionRun],
    *,
    outdir: str | Path,
    basename: str,
    write_jsonl: bool = True,
) -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}
    for run in runs:
        tag = f"{int(round(run.ratio * 100))}pct"
        txt_path = out / f"{basename}.{tag}.txt"
        txt_path.write_text(run.output_text, encoding="utf-8")
        paths[tag] = str(txt_path)

    if write_jsonl:
        meta_path = out / f"{basename}.meta.jsonl"
        with meta_path.open("w", encoding="utf-8") as f:
            for run in runs:
                rec = asdict(run)
                rec["output_text"] = None
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        paths["meta"] = str(meta_path)

    return paths

