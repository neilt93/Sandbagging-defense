"""Phase 4: extract per-stimulus hidden states from a HF causal LM.

For each stimulus row we:
  1. Build a "prefix-only" prompt — the comma-separated sequence WITHOUT
     any A/B answer scaffolding. This rules out the Cantos failure mode
     where a probe trained on activations that are conditioned on the
     answer-token region picks up output-derived information rather than
     internal structure.
  2. Run a single forward pass with output_hidden_states=True.
  3. Read the activation at the final stimulus token at every k-th
     layer (default stride 5).
  4. Save activations as a memmapped numpy array
     `results/activations/{model}/layers.npy` shape (N, n_layers, dim),
     with a sidecar parquet of (qid, role, family, params, layer_indices).

Real model runs require torch + transformers. This module's `extract`
function accepts an injectable `forward_fn` that returns hidden states
given a list of prompts; the test suite uses a synthetic forward_fn.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    layer_stride: int = 5
    prefix_only: bool = True
    batch_size: int = 4


ForwardFn = Callable[[list[str]], np.ndarray]
"""A function that takes a batch of prompts and returns activations
shape (batch, n_layers, hidden_dim) at the final input token."""


def build_prefix_prompt(sequence: list[int]) -> str:
    """Prefix-only prompt: comma-separated sequence with no answer label."""
    return ", ".join(str(t) for t in sequence)


def extract(
    stimuli_df: pl.DataFrame,
    forward_fn: ForwardFn,
    config: ExtractionConfig,
    output_dir: Path,
    model_name: str,
) -> dict[str, Path]:
    """Run forward_fn over every stimulus and save activations + sidecar.

    forward_fn is expected to:
      - take a list[str] of prompts
      - return ndarray (batch, n_layers, hidden_dim) at the final token

    We strided-subsample layers AFTER the call (so model code is simple).
    """
    out_dir = output_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = [build_prefix_prompt(seq) for seq in stimuli_df["sequence"].to_list()]
    n = len(prompts)

    # Prime: run a single prompt to discover shapes.
    sample = forward_fn([prompts[0]])
    n_layers_total = sample.shape[1]
    hidden_dim = sample.shape[2]
    layer_indices = list(range(0, n_layers_total, config.layer_stride))
    n_layers = len(layer_indices)

    arr_path = out_dir / "layers.npy"
    arr = np.memmap(arr_path, dtype=np.float32, mode="w+", shape=(n, n_layers, hidden_dim))

    # First sample row.
    arr[0] = sample[0, layer_indices, :].astype(np.float32)

    for start in range(1, n, config.batch_size):
        end = min(start + config.batch_size, n)
        batch = forward_fn(prompts[start:end])
        arr[start:end] = batch[:, layer_indices, :].astype(np.float32)

    arr.flush()
    sidecar = stimuli_df.select(["quadruplet_id", "role", "family", "params"]).with_columns(
        pl.lit(model_name).alias("model")
    )
    sidecar_path = out_dir / "sidecar.parquet"
    sidecar.write_parquet(sidecar_path)

    meta_path = out_dir / "meta.json"
    import json as _json

    meta_path.write_text(
        _json.dumps(
            {
                "model": model_name,
                "n": n,
                "n_layers_kept": n_layers,
                "n_layers_total": n_layers_total,
                "layer_stride": config.layer_stride,
                "layer_indices": layer_indices,
                "hidden_dim": hidden_dim,
                "prefix_only": config.prefix_only,
            }
        )
    )
    return {"layers": arr_path, "sidecar": sidecar_path, "meta": meta_path}


def hf_forward_fn(model_id: str, system_prompt: str | None = None) -> ForwardFn:
    """Construct an HF-backed forward_fn. Lazy-imports torch/transformers."""

    state: dict = {"loaded": False}

    def _ensure() -> None:
        if state["loaded"]:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map=device, output_hidden_states=True
        )
        model.eval()
        state.update(loaded=True, tok=tok, model=model, torch=torch, device=device)

    def forward(prompts: list[str]) -> np.ndarray:
        _ensure()
        torch = state["torch"]
        tok = state["tok"]
        model = state["model"]
        if system_prompt and hasattr(tok, "apply_chat_template"):
            prompts = [
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": p},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for p in prompts
            ]
        ids = tok(prompts, return_tensors="pt", padding=True).to(state["device"])
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        # tuple of (n_layers+1) tensors of shape (B, T, D); index final token.
        hs = torch.stack(out.hidden_states, dim=1)  # (B, n_layers+1, T, D)
        last_pos = ids.attention_mask.sum(dim=1) - 1
        gathered = hs[torch.arange(hs.shape[0]), :, last_pos, :]  # (B, n_layers+1, D)
        return gathered.float().cpu().numpy()

    return forward
