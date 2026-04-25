"""Phase 3: drive the sandbagging experiment.

For each (model, condition) pair, run:
  - MMLU (1000 sampled questions by default)
  - the Phase 2 preference harness on the same quadruplets

HumanEval is in the plan but requires generation + sandbox execution; it
is intentionally deferred to a follow-up to keep the Phase 3 surface
compact. The decision gate (preference accuracy holds while MMLU drops
≥ 20%) can be evaluated from MMLU + preference alone.

Outputs:
  results/sandbag/{model}/{condition}/mmlu.parquet
  results/sandbag/{model}/{condition}/preferences/{reframing}.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import polars as pl
from omegaconf import OmegaConf

from sandbagging_defense.elicitation.backends import get_backend
from sandbagging_defense.elicitation.harness import run_harness
from sandbagging_defense.sandbagging import mmlu
from sandbagging_defense.sandbagging.conditions import Condition, load_conditions

logger = logging.getLogger(__name__)


def run_one(
    backend_spec: dict[str, Any],
    condition: Condition,
    stimuli_df: pl.DataFrame,
    reframings: list[dict],
    mmlu_questions: list[dict],
    output_dir: Path,
    n_quadruplets: int | None,
    include_answer_marker: bool,
    progress: bool = True,
) -> dict[str, Path]:
    """Run a single (backend, condition) cell. Returns paths to outputs."""
    spec = {**backend_spec, "system_prompt": condition.system_prompt}
    backend = get_backend(spec)
    cell_dir = output_dir / backend.name / condition.id
    cell_dir.mkdir(parents=True, exist_ok=True)

    # MMLU
    mmlu_df = mmlu.evaluate(backend, mmlu_questions, include_answer_marker=include_answer_marker)
    mmlu_df = mmlu_df.with_columns(
        pl.lit(backend.name).alias("model"), pl.lit(condition.id).alias("condition")
    )
    mmlu_path = cell_dir / "mmlu.parquet"
    mmlu_df.write_parquet(mmlu_path)
    logger.info(
        "%s/%s mmlu accuracy = %.3f (n=%d)",
        backend.name,
        condition.id,
        mmlu.accuracy(mmlu_df),
        mmlu_df.height,
    )

    # Preferences
    pref_dir = cell_dir / "preferences"
    pref_paths = run_harness(
        backend=backend,
        stimuli_df=stimuli_df,
        reframings=reframings,
        output_dir=pref_dir,
        n_quadruplets=n_quadruplets,
        progress=progress,
    )

    backend.close()
    return {"mmlu": mmlu_path, **{f"pref_{k}": v for k, v in pref_paths.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend-kind", choices=["mock", "hf", "vllm"], default="vllm")
    parser.add_argument("--n-quadruplets", type=int, default=None)
    parser.add_argument(
        "--mmlu-jsonl",
        type=Path,
        default=None,
        help="Local JSONL for MMLU (skips HF datasets dependency)",
    )
    parser.add_argument(
        "--include-answer-marker",
        action="store_true",
        help="Used only by the mock backend pipeline test path",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = OmegaConf.load(args.config)
    pilot_cfg = OmegaConf.load(cfg.pilot_config_path)
    reframings_cfg = OmegaConf.load(pilot_cfg.reframings_path)
    reframings = [dict(r) for r in reframings_cfg.reframings]

    stimuli_df = pl.read_parquet(pilot_cfg.stimuli_path)
    conditions = load_conditions(args.config)

    if args.mmlu_jsonl:
        mmlu_questions = mmlu.load_jsonl(args.mmlu_jsonl)
    else:
        mmlu_questions = mmlu.load_hf_mmlu(
            n=cfg.benchmarks.mmlu.n_questions, seed=cfg.benchmarks.mmlu.seed
        )

    output_root = Path(cfg.output_dir)
    for model in pilot_cfg.models:
        backend_spec = {
            "kind": args.backend_kind,
            "name": model.name,
            "hf_id": model.hf_id,
        }
        for condition in conditions:
            run_one(
                backend_spec=backend_spec,
                condition=condition,
                stimuli_df=stimuli_df,
                reframings=reframings,
                mmlu_questions=mmlu_questions,
                output_dir=output_root,
                n_quadruplets=args.n_quadruplets,
                include_answer_marker=args.include_answer_marker,
                progress=not args.no_progress,
            )


if __name__ == "__main__":
    main()
