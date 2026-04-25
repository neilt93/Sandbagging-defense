"""Phase 2 pipeline tests, using the mock backend.

Verifies that the harness wires correctly end-to-end:
  - prompt formatting injects A/B sequences into reframing templates,
  - all 6 unordered pairs in both orders are scored,
  - position balancing collapses 12 ordered rows into 6 balanced ones,
  - the analysis pipeline recovers the synthetic preference signal,
  - parquet output schema is stable.

No real model is loaded. The mock backend uses a deterministic scorer
that prefers sequences with lower LZ76 phrase counts, so the pipeline
should recover a positive Spearman between LZ gap and preference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sandbagging_defense.analysis.pilot import (
    attach_lz_gap,
    cross_reframing_agreement,
    load_pilot,
    lz_rank_correlation,
    preference_accuracy,
    scaling_table,
)
from sandbagging_defense.elicitation.backends import (
    HFBackend,
    MockBackend,
    PairScore,
    VLLMBackend,
    get_backend,
)
from sandbagging_defense.elicitation.harness import (
    build_prompt,
    format_sequence,
    position_balanced_preferences,
    run_harness,
    score_quadruplet,
)
from sandbagging_defense.stimuli.generators import generate_corpus
from sandbagging_defense.stimuli.quadruplets import ROLES

# --------------------------------------------------------------------------- #
# Backend factory and primitives
# --------------------------------------------------------------------------- #


def test_get_backend_dispatch() -> None:
    assert isinstance(get_backend({"kind": "mock", "name": "mock"}), MockBackend)
    assert isinstance(
        get_backend({"kind": "hf", "name": "tiny", "hf_id": "sshleifer/tiny-gpt2"}), HFBackend
    )
    assert isinstance(
        get_backend({"kind": "vllm", "name": "tiny", "hf_id": "anything"}), VLLMBackend
    )
    with pytest.raises(ValueError):
        get_backend({"kind": "unknown"})


def test_pair_score_prob_a() -> None:
    s = PairScore(logp_a=-1.0, logp_b=-2.0)
    assert 0.7 < s.prob_a < 0.74  # softmax(-1,-2) ≈ (0.731, 0.269)


def test_format_and_build_prompt() -> None:
    seq_a = [1, 2, 3]
    seq_b = [4, 5, 6]
    template = "A: {a}\nB: {b}"
    p = build_prompt(template, seq_a, seq_b)
    assert format_sequence(seq_a) in p
    assert format_sequence(seq_b) in p
    assert p.startswith("A: 1, 2, 3\nB: 4, 5, 6")


# --------------------------------------------------------------------------- #
# Mock backend: synthetic preference matches LZ structure direction
# --------------------------------------------------------------------------- #


def test_mock_backend_prefers_more_structured() -> None:
    backend = MockBackend()
    structured = [0, 0, 0, 0, 0, 0]  # zero transitions
    shuffled = [0, 1, 2, 3, 4, 5]  # max transitions
    template = "A: {a}\nB: {b}"
    s = backend.score_pair(build_prompt(template, structured, shuffled), "A", "B")
    assert s.prob_a > 0.5
    s_swap = backend.score_pair(build_prompt(template, shuffled, structured), "A", "B")
    assert s_swap.prob_a < 0.5


def test_score_quadruplet_returns_12_ordered_rows() -> None:
    backend = MockBackend()
    quad = {
        "structured": [0] * 16,
        "shuffled": [0, 1] * 8,
        "markov_pastiche": [0, 0, 1, 1] * 4,
        "random_match": [0, 1, 2, 3, 4, 5, 6, 7] * 2,
    }
    rows = score_quadruplet(backend, quad, template="A: {a}\nB: {b}")
    assert len(rows) == 12  # C(4,2) * 2
    seen_unordered = {tuple(sorted((r["left_role"], r["right_role"]))) for r in rows}
    assert len(seen_unordered) == 6


def test_position_balancing_collapses_to_6() -> None:
    rows = [
        {"left_role": "structured", "right_role": "shuffled", "prob_a": 0.7},
        {"left_role": "shuffled", "right_role": "structured", "prob_a": 0.4},
        {"left_role": "structured", "right_role": "markov_pastiche", "prob_a": 0.6},
        {"left_role": "markov_pastiche", "right_role": "structured", "prob_a": 0.5},
        {"left_role": "structured", "right_role": "random_match", "prob_a": 0.8},
        {"left_role": "random_match", "right_role": "structured", "prob_a": 0.3},
        {"left_role": "shuffled", "right_role": "markov_pastiche", "prob_a": 0.45},
        {"left_role": "markov_pastiche", "right_role": "shuffled", "prob_a": 0.55},
        {"left_role": "shuffled", "right_role": "random_match", "prob_a": 0.5},
        {"left_role": "random_match", "right_role": "shuffled", "prob_a": 0.5},
        {"left_role": "markov_pastiche", "right_role": "random_match", "prob_a": 0.6},
        {"left_role": "random_match", "right_role": "markov_pastiche", "prob_a": 0.4},
    ]
    balanced = position_balanced_preferences(rows)
    assert len(balanced) == 6
    # structured should be preferred over shuffled: avg(0.7, 1-0.4) = 0.65
    sb = next(b for b in balanced if {b["role_x"], b["role_y"]} == {"structured", "shuffled"})
    target = sb["prob_x_over_y"] if sb["role_x"] == "structured" else 1 - sb["prob_x_over_y"]
    assert abs(target - 0.65) < 1e-9


def test_position_balancing_corrects_constant_bias() -> None:
    """If the model has a constant 'always prefer left' bias, balancing
    cancels it: the balanced preference for an indistinguishable pair is 0.5.
    """
    rows = [
        {"left_role": "structured", "right_role": "shuffled", "prob_a": 0.9},
        {"left_role": "shuffled", "right_role": "structured", "prob_a": 0.9},
    ]
    rows += [
        # other 5 pairs stubbed
        *[
            {
                "left_role": x,
                "right_role": y,
                "prob_a": 0.5,
            }
            for x, y in [
                ("structured", "markov_pastiche"),
                ("markov_pastiche", "structured"),
                ("structured", "random_match"),
                ("random_match", "structured"),
                ("shuffled", "markov_pastiche"),
                ("markov_pastiche", "shuffled"),
                ("shuffled", "random_match"),
                ("random_match", "shuffled"),
                ("markov_pastiche", "random_match"),
                ("random_match", "markov_pastiche"),
            ]
        ]
    ]
    balanced = position_balanced_preferences(rows)
    sb = next(b for b in balanced if {b["role_x"], b["role_y"]} == {"structured", "shuffled"})
    target = sb["prob_x_over_y"] if sb["role_x"] == "structured" else 1 - sb["prob_x_over_y"]
    assert abs(target - 0.5) < 1e-9


# --------------------------------------------------------------------------- #
# End-to-end: run_harness + analysis recovers the synthetic signal
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def small_corpus() -> pl.DataFrame:
    rows = list(generate_corpus(n_quadruplets=40, length=128, vocab=16, seed=0))
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def reframings() -> list[dict]:
    return [
        {
            "id": "structure",
            "template": "A: {a}\nB: {b}\nWhich shows more structure? Answer A or B.",
        },
        {
            "id": "predictability",
            "template": "A: {a}\nB: {b}\nWhich is easier to predict? Answer A or B.",
        },
    ]


def test_run_harness_writes_parquet(tmp_path: Path, small_corpus, reframings) -> None:
    backend = MockBackend(name="mock-test")
    out = run_harness(
        backend=backend,
        stimuli_df=small_corpus,
        reframings=reframings,
        output_dir=tmp_path,
        n_quadruplets=10,
        progress=False,
    )
    assert set(out.keys()) == {"structure", "predictability"}
    for path in out.values():
        df = pl.read_parquet(path)
        assert df.height == 10 * 6  # 6 balanced pairs per quadruplet
        assert set(df.columns) >= {
            "model",
            "reframing",
            "quadruplet_id",
            "family",
            "role_x",
            "role_y",
            "prob_x_over_y",
        }
        assert df["prob_x_over_y"].min() >= 0.0
        assert df["prob_x_over_y"].max() <= 1.0


def test_attach_lz_gap_and_recover_signal(tmp_path: Path, small_corpus, reframings) -> None:
    backend = MockBackend(name="mock-test")
    out_dir = tmp_path / "pilot"
    run_harness(
        backend=backend,
        stimuli_df=small_corpus,
        reframings=reframings,
        output_dir=out_dir / backend.name,
        n_quadruplets=20,
        progress=False,
    )

    # load_pilot expects pilot_dir/{model}/{reframing}.parquet
    df = load_pilot(out_dir)
    assert df.height == 20 * 6 * 2

    df_with_lz = attach_lz_gap(df, small_corpus)
    assert "lz_gap" in df_with_lz.columns
    assert df_with_lz["lz_gap"].is_not_null().all()

    # Mock prefers fewer transitions; lower-LZ side should be preferred.
    acc = preference_accuracy(df_with_lz)
    assert acc.height >= 1
    assert (acc["accuracy"] >= 0.5).all(), acc

    corr = lz_rank_correlation(df_with_lz)
    # Mock backend produces a positive structure-tracking signal.
    assert corr.height >= 1
    assert (corr["spearman_r"] > 0).all(), corr


def test_cross_reframing_agreement_high_for_deterministic_mock(
    tmp_path: Path, small_corpus, reframings
) -> None:
    # The mock scorer doesn't read the prompt's question wording, so the
    # two reframings produce IDENTICAL preferences -> Spearman = 1.
    backend = MockBackend(name="mock-test")
    out_dir = tmp_path / "pilot"
    run_harness(
        backend=backend,
        stimuli_df=small_corpus,
        reframings=reframings,
        output_dir=out_dir / backend.name,
        n_quadruplets=15,
        progress=False,
    )
    df = load_pilot(out_dir)
    df = attach_lz_gap(df, small_corpus)
    cross = cross_reframing_agreement(df)
    assert cross.height == 1
    # Allow for ties / numerical fuzz; mock is deterministic so it should be ~1.
    assert cross["spearman_r"][0] > 0.999


def test_scaling_table_attaches_sizes(tmp_path: Path, small_corpus, reframings) -> None:
    # Two pseudo-models with different size labels.
    out_dir = tmp_path / "pilot"
    for name in ("mock-small", "mock-big"):
        b = MockBackend(name=name)
        run_harness(
            backend=b,
            stimuli_df=small_corpus,
            reframings=reframings[:1],
            output_dir=out_dir / name,
            n_quadruplets=5,
            progress=False,
        )
    df = load_pilot(out_dir)
    df = attach_lz_gap(df, small_corpus)
    acc = preference_accuracy(df)
    sizes = {"mock-small": 0.5, "mock-big": 32.0}
    table = scaling_table(acc, sizes)
    assert "size_b" in table.columns
    sizes_out = table["size_b"].to_list()
    assert min(sizes_out) == 0.5
    assert max(sizes_out) == 32.0


def test_roles_complete() -> None:
    # Sanity: ROLES has the four expected roles in a known order.
    assert set(ROLES) == {"structured", "shuffled", "markov_pastiche", "random_match"}


def test_lz_rank_correlation_handles_constant_input() -> None:
    """Edge case: if all lz_gaps are equal, Spearman is undefined; the
    function should still return a row (with NaN or warning) without crashing.
    """
    df = pl.DataFrame(
        {
            "model": ["m"] * 6,
            "reframing": ["r"] * 6,
            "quadruplet_id": [0] * 6,
            "role_x": ["a"] * 6,
            "role_y": ["b"] * 6,
            "prob_x_over_y": np.linspace(0.0, 1.0, 6),
            "lz_gap": [0.0] * 6,
        }
    )
    out = lz_rank_correlation(df)
    # constant lz_gap -> nan correlation but no crash
    assert out.height == 1
