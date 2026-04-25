"""Tests for the figures module: synthetic demo and real-data render path."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from sandbagging_defense.analysis.figures import (
    lz_preference_scatter,
    money_plot,
    synthetic_money_demo,
)


def test_synthetic_money_demo_shape() -> None:
    df = synthetic_money_demo()
    assert df.height > 0
    assert set(df.columns) >= {"model", "condition", "metric", "accuracy", "relative_perf"}
    # Normal baseline = 1.0 for every (model, metric).
    normal = df.filter(pl.col("condition") == "normal")
    assert (normal["relative_perf"] == 1.0).all()


def test_money_plot_writes_pdf(tmp_path: Path) -> None:
    df = synthetic_money_demo()
    out = tmp_path / "money.pdf"
    money_plot(df, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_money_plot_empty_raises(tmp_path: Path) -> None:
    df = pl.DataFrame(
        schema={
            "model": pl.Utf8,
            "condition": pl.Utf8,
            "metric": pl.Utf8,
            "accuracy": pl.Float64,
            "relative_perf": pl.Float64,
        }
    )
    with pytest.raises(ValueError):
        money_plot(df, tmp_path / "x.pdf")


def test_lz_preference_scatter_writes_pdf(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "model": ["m"] * 12,
            "reframing": ["r1"] * 6 + ["r2"] * 6,
            "quadruplet_id": list(range(6)) * 2,
            "role_x": ["structured"] * 12,
            "role_y": ["shuffled"] * 12,
            "prob_x_over_y": [0.7, 0.6, 0.55, 0.5, 0.4, 0.3] * 2,
            "lz_gap": [10, 5, 2, 0, -3, -8] * 2,
        }
    )
    out = tmp_path / "scatter.pdf"
    lz_preference_scatter(df, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_lz_preference_scatter_requires_lz_gap(tmp_path: Path) -> None:
    df = pl.DataFrame({"model": ["m"], "prob_x_over_y": [0.5]})
    with pytest.raises(ValueError, match="lz_gap"):
        lz_preference_scatter(df, tmp_path / "x.pdf")
