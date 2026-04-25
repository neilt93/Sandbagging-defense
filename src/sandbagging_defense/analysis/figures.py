"""Regenerate paper figures from results/.

Two main figures:
  - money plot: x = condition, y = relative_perf vs normal baseline,
    hue = metric (mmlu vs preference), one panel per model
  - LZ-vs-preference scatter (Phase 2): per (model, reframing),
    scatter of lz_gap vs prob_x_over_y - 0.5

When real result directories are missing, falls back to a synthetic
demo run so the figure pipeline is testable end-to-end and the user
can verify the plot will render before kicking off the GPU pilot.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402
import seaborn as sns  # noqa: E402

logger = logging.getLogger(__name__)


def money_plot(money_df: pl.DataFrame, out_path: Path, title: str | None = None) -> None:
    """Render the central money plot to `out_path` (PDF).

    Expected columns: model, condition, metric, accuracy, relative_perf.
    """
    if money_df.height == 0:
        raise ValueError("money_df is empty; nothing to plot")

    pdf = money_df.to_pandas()
    sns.set_theme(style="whitegrid", context="paper")
    models = pdf["model"].unique()
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        sub = pdf[pdf["model"] == model]
        sns.lineplot(
            data=sub,
            x="condition",
            y="relative_perf",
            hue="metric",
            marker="o",
            ax=ax,
        )
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(0.5, color="lightgray", linestyle=":", linewidth=0.6)
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(model)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        if ax is axes[0, 0]:
            ax.set_ylabel("relative performance vs `normal`")
        else:
            ax.set_ylabel("")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def lz_preference_scatter(pref_df: pl.DataFrame, out_path: Path, title: str | None = None) -> None:
    """Per (model, reframing) scatter of lz_gap vs signed preference."""
    if "lz_gap" not in pref_df.columns:
        raise ValueError("pref_df must already have lz_gap attached (Phase 2 analysis)")
    if pref_df.height == 0:
        raise ValueError("pref_df is empty; nothing to plot")
    pdf = pref_df.to_pandas()
    pdf["signed_pref"] = pdf["prob_x_over_y"] - 0.5

    sns.set_theme(style="whitegrid", context="paper")
    g = sns.relplot(
        data=pdf,
        x="lz_gap",
        y="signed_pref",
        col="reframing",
        row="model",
        kind="scatter",
        alpha=0.4,
        height=2.6,
    )
    g.refline(y=0, color="gray", linestyle=":")
    g.refline(x=0, color="gray", linestyle=":")
    g.set_axis_labels("LZ gap (lz_y - lz_x)", "preference for x (prob - 0.5)")
    if title:
        g.fig.suptitle(title, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out_path, bbox_inches="tight")
    plt.close(g.fig)
    logger.info("wrote %s", out_path)


def synthetic_money_demo() -> pl.DataFrame:
    """Build a synthetic money-plot frame: MMLU collapses, preference holds."""
    rows = []
    for model_name, baseline_mmlu in [
        ("qwen2.5-0.5b", 0.40),
        ("qwen2.5-7b", 0.70),
        ("qwen2.5-32b", 0.82),
    ]:
        for cond, mmlu_drop, pref_drop in [
            ("normal", 0.00, 0.00),
            ("sandbag_targeted", 0.55, 0.05),
            ("sandbag_general", 0.40, 0.03),
        ]:
            rows.append(
                {
                    "model": model_name,
                    "condition": cond,
                    "metric": "mmlu",
                    "accuracy": baseline_mmlu * (1 - mmlu_drop),
                    "relative_perf": 1 - mmlu_drop,
                }
            )
            rows.append(
                {
                    "model": model_name,
                    "condition": cond,
                    "metric": "preference",
                    "accuracy": 0.78 * (1 - pref_drop),
                    "relative_perf": 1 - pref_drop,
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output directory for PDFs")
    parser.add_argument(
        "--sandbag-dir",
        type=Path,
        default=Path("results/sandbag"),
        help="Sandbag results dir (Phase 3)",
    )
    parser.add_argument(
        "--stimuli",
        type=Path,
        default=Path("results/stimuli/v1.parquet"),
        help="Stimuli parquet (Phase 1)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Render a synthetic money plot when no real results exist",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)

    if args.demo or not args.sandbag_dir.exists():
        if not args.demo:
            print(
                f"sandbag dir {args.sandbag_dir} not found; rendering synthetic demo. "
                "Pass --demo to silence this.",
                file=sys.stderr,
            )
        money_df = synthetic_money_demo()
        money_plot(money_df, args.out / "money_plot_demo.pdf", title="DEMO (synthetic)")
        return

    # Real data path: lazy-imported so the demo path stays light.
    from sandbagging_defense.analysis.pilot import attach_lz_gap
    from sandbagging_defense.analysis.sandbag import (
        load_sandbag_mmlu,
        load_sandbag_preferences,
        mmlu_summary,
        money_plot_frame,
        preference_summary,
    )

    stimuli_df = pl.read_parquet(args.stimuli)
    mmlu_df = load_sandbag_mmlu(args.sandbag_dir)
    pref_df = load_sandbag_preferences(args.sandbag_dir)
    if mmlu_df.height == 0 or pref_df.height == 0:
        raise SystemExit(f"no mmlu/preference rows under {args.sandbag_dir}")

    money = money_plot_frame(mmlu_summary(mmlu_df), preference_summary(pref_df, stimuli_df))
    money_plot(money, args.out / "money_plot.pdf")

    pref_with_lz = attach_lz_gap(pref_df, stimuli_df)
    lz_preference_scatter(pref_with_lz, args.out / "lz_preference_scatter.pdf")


if __name__ == "__main__":
    main()
