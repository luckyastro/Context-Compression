from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ICAEConfig:
    """
    Minimal, compression-only ICAE loader.

    Required deps (not installed by default in this repo):
    - torch
    - transformers
    - peft
    - safetensors

    You must also have:
    - base model weights accessible for `base_model_name_or_path`
    - ICAE checkpoint in safetensors format containing LoRA + memory embeddings
    """

    base_model_name_or_path: str
    checkpoint_safetensors_path: str
    device: str = "auto"  # auto|cpu|cuda|cuda:0...
    bf16: bool = False
    fixed_mem_size: int = 128
    mean_compression_rate: int = 4
    max_input_tokens: int = 5120
    max_generate_tokens: int = 512

    # Low-VRAM options (require bitsandbytes when using 4-bit/8-bit)
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    device_map: str = "none"  # none | auto (auto enables HF dispatch/offload)
    offload_folder: str = "offload"

    # LoRA config (must match how checkpoint was trained)
    lora_r: int = 512
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class ICAECompressor:
    """
    Compression-only ICAE.

    This is adapted from the earlier ICAE clone but stripped down to:
    - model init
    - `_compress()` to memory slots
    - `compress_generate()` to decode to text
    """

    def __init__(self, cfg: ICAEConfig):
        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore
            from safetensors.torch import load_file  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            from peft import LoraConfig, get_peft_model  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "ICAE stage requires torch/transformers/peft/safetensors. "
                "Install them to use --icae-* options.\n\n"
                "Suggested installs:\n"
                "- CPU:\n"
                "  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "  python -m pip install transformers peft safetensors\n"
                "- CUDA (example for CUDA 12.1 wheels):\n"
                "  python -m pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
                "  python -m pip install transformers peft safetensors\n"
            ) from e

        self._torch = torch
        self._nn = nn
        self._load_file = load_file
        self._AutoModelForCausalLM = AutoModelForCausalLM
        self._AutoTokenizer = AutoTokenizer
        self._LoraConfig = LoraConfig
        self._get_peft_model = get_peft_model

        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)

        dtype = torch.float16 if not cfg.bf16 else torch.bfloat16

        # Low VRAM loading (quantization / device_map offload).
        # Note: if device_map != "none", we should NOT call .to(device).
        model_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }

        if cfg.device_map and cfg.device_map != "none":
            model_kwargs["device_map"] = cfg.device_map
            model_kwargs["offload_folder"] = cfg.offload_folder

        if cfg.load_in_4bit or cfg.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "ICAE quantized loading requires transformers + bitsandbytes.\n"
                    "Install bitsandbytes, or disable --icae-load-in-4bit/8bit."
                ) from e

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=cfg.load_in_4bit,
                load_in_8bit=cfg.load_in_8bit,
                bnb_4bit_compute_dtype=(torch.bfloat16 if cfg.bf16 else torch.float16),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = bnb_cfg

        base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, **model_kwargs)

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        base = get_peft_model(base, lora_config)

        self.icae = base if (cfg.device_map and cfg.device_map != "none") else base.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path, use_fast=False)

        self.vocab_size = self.icae.config.vocab_size + 1  # + [PAD]
        self.pad_token_id = self.vocab_size - 1
        self.mem_size = cfg.fixed_mem_size
        self.mean_compression_rate = cfg.mean_compression_rate

        self.vocab_size_with_mem = self.vocab_size + self.mem_size
        self.ae_token_id = self.vocab_size_with_mem + 0
        # (lm_token_id/ft_token_id omitted; not needed for compression-only decoding)

        # Resize token embeddings for mem tokens + special token
        self.icae.resize_token_embeddings(self.vocab_size_with_mem + 1)
        self.eos_id = 2  # llama/mistral tokenizers typically use 2

        self.dim = self.icae.config.hidden_size
        self.memory_token_embed = nn.Embedding(self.mem_size + 1, self.dim, padding_idx=None).to(self.device)

        self.append_sequence = torch.arange(
            self.vocab_size,
            self.vocab_size + self.mem_size,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        # Load checkpoint (LoRA + memory embeddings). strict=False because base weights are not in checkpoint.
        state = load_file(cfg.checkpoint_safetensors_path)
        self.icae.load_state_dict(state, strict=False)
        self.eval()

    def _resolve_device(self, device: str) -> str:
        torch = self._torch
        d = (device or "auto").strip().lower()
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def eval(self) -> None:
        self.icae.eval()

    def _compute_num_segments(self, total_length: int) -> int:
        return math.ceil(total_length / (self.mem_size * self.mean_compression_rate))

    def _tokens_to_embeddings(self, token_ids):
        torch = self._torch
        embs = self.icae.get_base_model().model.embed_tokens(token_ids)
        special = token_ids >= self.vocab_size
        if special.any():
            embs[special] = self.memory_token_embed((token_ids[special] - self.vocab_size)).to(embs)
        return embs

    def _compress(self, input_ids):
        torch = self._torch

        total_length = input_ids.size(1)
        num_segments = self._compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)

        max_compressed_length = num_segments * self.mem_size
        compress_outputs = torch.zeros((max_compressed_length, self.dim), device=self.device, dtype=self.icae.dtype)

        for segment_idx in range(num_segments):
            start = segment_idx * segment_length
            end = min((segment_idx + 1) * segment_length, total_length)
            seg_ids = input_ids[:, start:end]
            seg_ids = torch.cat([seg_ids, self.append_sequence], dim=1)
            mem_flag = seg_ids >= self.vocab_size

            seg_embs = self.icae.get_base_model().model.embed_tokens(seg_ids)
            if mem_flag.any():
                seg_embs[mem_flag] = self.memory_token_embed((seg_ids[mem_flag] - self.vocab_size)).to(seg_embs)

            out = self.icae(inputs_embeds=seg_embs, output_hidden_states=True)
            hs = out.hidden_states[-1]
            compress_outputs[segment_idx * self.mem_size : (segment_idx + 1) * self.mem_size] = hs[mem_flag]

        return compress_outputs

    def compress_generate(self, text: str) -> str:
        torch = self._torch

        tok = self.tokenizer(
            text,
            truncation=True,
            max_length=self.cfg.max_input_tokens,
            padding=False,
            return_attention_mask=False,
        )
        input_ids = torch.LongTensor([tok["input_ids"]]).to(self.device)
        memory_slots = self._compress(input_ids)

        prompt_ids = torch.LongTensor([[self.ae_token_id]]).to(self.device)
        prompt_embs = self._tokens_to_embeddings(prompt_ids)
        memory_slots = memory_slots.to(prompt_embs)

        decoder_input = torch.cat((memory_slots.unsqueeze(0), prompt_embs), dim=1)
        output = decoder_input.clone()

        past_key_values = None
        gen_ids = []
        with torch.no_grad():
            for _ in range(self.cfg.max_generate_tokens):
                with self.icae.disable_adapter():
                    out = self.icae(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)
                logit = out.logits[:, -1, : self.vocab_size - 1]
                past_key_values = out.past_key_values
                next_id = torch.argmax(logit, dim=-1)
                if next_id.item() == self.eos_id:
                    break
                output = self.icae.get_base_model().model.embed_tokens(next_id).unsqueeze(1).to(self.device)
                gen_ids.append(next_id.item())

        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

