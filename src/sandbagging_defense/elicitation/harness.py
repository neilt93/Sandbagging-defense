"""Phase 2: pairwise preference harness.

For each quadruplet (4 stimuli), enumerate all 6 unordered pairs in both
positional orders -> 12 calls per quadruplet per reframing. Three
reframings -> 36 calls per quadruplet per model. Per-model, per-reframing
results are written to results/pilot/{model}/{reframing}.parquet.

Each row records the raw A/B logprobs and the position-balanced
preference probability. Position balancing averages prob_a from the
(A,B) order with prob_b from the (B,A) order:

    p_balanced = 0.5 * (prob_a_in_AB + prob_b_in_BA)

This isolates the structural preference from the model's positional bias.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl
from omegaconf import OmegaConf
from tqdm import tqdm

from sandbagging_defense.elicitation.backends import Backend, PairScore, get_backend
from sandbagging_defense.stimuli.quadruplets import ROLES

logger = logging.getLogger(__name__)


def format_sequence(seq: list[int]) -> str:
    """Format a token sequence for prompt insertion: comma-separated."""
    return ", ".join(str(t) for t in seq)


def build_prompt(template: str, a_seq: list[int], b_seq: list[int]) -> str:
    return template.format(a=format_sequence(a_seq), b=format_sequence(b_seq))


def score_quadruplet(
    backend: Backend,
    quad: dict[str, list[int]],
    template: str,
    a_token: str = "A",
    b_token: str = "B",
) -> list[dict]:
    """Score all 6 ordered pairs in both positional orders.

    Returns 12 rows: one per (left_role, right_role) ordering. Each row
    has prob_a (probability the model picks the LEFT side).
    """
    rows: list[dict] = []
    pairs = list(itertools.combinations(ROLES, 2))
    for left, right in pairs:
        for a_role, b_role in [(left, right), (right, left)]:
            prompt = build_prompt(template, quad[a_role], quad[b_role])
            score: PairScore = backend.score_pair(prompt, a_token, b_token)
            rows.append(
                {
                    "left_role": a_role,
                    "right_role": b_role,
                    "logp_a": score.logp_a,
                    "logp_b": score.logp_b,
                    "prob_a": score.prob_a,
                }
            )
    return rows


def position_balanced_preferences(rows: list[dict]) -> list[dict]:
    """Collapse the 12 ordered rows into 6 position-balanced preferences.

    For each unordered pair {X, Y}, returns prob(X over Y) averaged over
    both positional orders.
    """
    by_unordered: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = tuple(sorted((r["left_role"], r["right_role"])))
        by_unordered.setdefault(key, []).append(r)

    balanced: list[dict] = []
    for (x, y), rs in by_unordered.items():
        # rs has 2 entries: one with left=x and one with left=y.
        prob_x_over_y_when_left = next(r["prob_a"] for r in rs if r["left_role"] == x)
        prob_x_over_y_when_right = 1.0 - next(r["prob_a"] for r in rs if r["left_role"] == y)
        p = 0.5 * (prob_x_over_y_when_left + prob_x_over_y_when_right)
        balanced.append({"role_x": x, "role_y": y, "prob_x_over_y": p})
    return balanced


def run_harness(
    backend: Backend,
    stimuli_df: pl.DataFrame,
    reframings: list[dict],
    output_dir: Path,
    n_quadruplets: int | None = None,
    progress: bool = True,
) -> dict[str, Path]:
    """Run the harness across all quadruplets x reframings.

    Returns a map {reframing_id: output_path}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    quad_ids = stimuli_df["quadruplet_id"].unique().sort()
    if n_quadruplets is not None:
        quad_ids = quad_ids[:n_quadruplets]

    out_paths: dict[str, Path] = {}
    for reframing in reframings:
        rid = reframing["id"]
        template = reframing["template"]
        rows: list[dict] = []
        iterator: Iterable = quad_ids
        if progress:
            iterator = tqdm(quad_ids, desc=f"reframing={rid}")
        for q_id in iterator:
            quad_rows = stimuli_df.filter(pl.col("quadruplet_id") == q_id).to_dicts()
            quad = {r["role"]: r["sequence"] for r in quad_rows}
            family = quad_rows[0]["family"]
            params = quad_rows[0]["params"]
            ordered = score_quadruplet(backend, quad, template)
            balanced = position_balanced_preferences(ordered)
            for r in balanced:
                rows.append(
                    {
                        "model": backend.name,
                        "reframing": rid,
                        "quadruplet_id": int(q_id),
                        "family": family,
                        "params": params,
                        **r,
                    }
                )
        out_path = output_dir / f"{rid}.parquet"
        pl.DataFrame(rows).write_parquet(out_path)
        out_paths[rid] = out_path
        logger.info("wrote %d rows to %s", len(rows), out_path)
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit number of quadruplets (for smoke runs)",
    )
    parser.add_argument(
        "--backend-kind",
        choices=["mock", "hf", "vllm"],
        default=None,
        help="Override the backend kind (default: vllm for production)",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = OmegaConf.load(args.config)
    reframings_cfg = OmegaConf.load(cfg.reframings_path)
    reframings: list[dict] = [dict(r) for r in reframings_cfg.reframings]

    stimuli_df = pl.read_parquet(cfg.stimuli_path)

    output_root = Path(cfg.output_dir)
    for model in cfg.models:
        spec: dict[str, Any] = {
            "kind": args.backend_kind or "vllm",
            "name": model.name,
            "hf_id": model.hf_id,
        }
        backend = get_backend(spec)
        run_harness(
            backend=backend,
            stimuli_df=stimuli_df,
            reframings=reframings,
            output_dir=output_root / model.name,
            n_quadruplets=args.n,
            progress=not args.no_progress,
        )
        backend.close()
    print(json.dumps({"output_dir": str(output_root)}))


if __name__ == "__main__":
    main()
