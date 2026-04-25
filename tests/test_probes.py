"""Phase 4 pipeline tests with synthetic activations.

We build activations where one fixed layer encodes the binary
'is-structured' label as a learnable signal, and verify:
  - extract.extract() correctly batches, slices layers, and saves
    memmapped arrays + sidecar
  - quadruplet_id_split assigns siblings consistently
  - sweep_layers recovers high AUROC at the encoding layer and
    near-chance elsewhere
  - levinstein_herrmann_transfer correctly trains/tests across families
  - cantos_compare returns 'internal' when prefix-only AUROC matches
    answer-conditioned AUROC, 'leak' when it does not
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sandbagging_defense.probes.extract import (
    ExtractionConfig,
    build_prefix_prompt,
    extract,
)
from sandbagging_defense.probes.train import (
    ProbeConfig,
    cantos_compare,
    label_structured_vs_control,
    levinstein_herrmann_transfer,
    load_activations,
    quadruplet_id_split,
    sweep_layers,
    train_probe_at_layer,
)
from sandbagging_defense.stimuli.generators import generate_corpus

# --------------------------------------------------------------------------- #
# Synthetic forward function: layer 2 encodes the label, others are noise
# --------------------------------------------------------------------------- #


def _make_synthetic_forward(
    label_layer: int = 2,
    n_layers: int = 6,
    hidden_dim: int = 32,
    label_strength: float = 5.0,
) -> tuple[callable, callable]:
    """Returns (forward_fn, set_labels_fn). Caller sets the per-prompt
    labels via set_labels_fn before each batch. Noise is keyed off the
    prompt's hash so calls are deterministic AND unique per prompt
    (without this, batched calls reset the same RNG state and fake
    structure leaks into non-encoding layers).
    """
    state: dict = {"prompts_to_labels": {}}

    def set_labels(mapping: dict[str, int]) -> None:
        state["prompts_to_labels"] = mapping

    import hashlib

    def _stable_seed(p: str) -> int:
        h = hashlib.md5(p.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "little", signed=False)

    def forward(prompts: list[str]) -> np.ndarray:
        out = np.zeros((len(prompts), n_layers, hidden_dim), dtype=np.float32)
        for i, p in enumerate(prompts):
            rng = np.random.default_rng(_stable_seed(p))
            label = state["prompts_to_labels"].get(p, 0)
            out[i] = rng.normal(0, 0.5, size=(n_layers, hidden_dim))
            out[i, label_layer, 0] += label_strength * (2 * label - 1)
        return out

    return forward, set_labels


# --------------------------------------------------------------------------- #
# Build a small corpus and synthetic activations
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def small_corpus() -> pl.DataFrame:
    rows = list(generate_corpus(n_quadruplets=30, length=128, vocab=16, seed=0))
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# extract.py
# --------------------------------------------------------------------------- #


def test_build_prefix_prompt_format() -> None:
    p = build_prefix_prompt([1, 2, 3, 4])
    assert p == "1, 2, 3, 4"


def test_extract_shapes_and_files(tmp_path: Path, small_corpus) -> None:
    forward, set_labels = _make_synthetic_forward(n_layers=10, hidden_dim=32)

    # Map each prompt to its true label.
    label_map: dict[str, int] = {}
    for row in small_corpus.iter_rows(named=True):
        label_map[build_prefix_prompt(row["sequence"])] = int(row["role"] == "structured")
    set_labels(label_map)

    out = extract(
        stimuli_df=small_corpus,
        forward_fn=forward,
        config=ExtractionConfig(layer_stride=2, prefix_only=True, batch_size=4),
        output_dir=tmp_path,
        model_name="synth",
    )
    assert out["layers"].exists()
    assert out["sidecar"].exists()
    assert out["meta"].exists()

    arr, sidecar, meta = load_activations(out["layers"], out["sidecar"], out["meta"])
    assert arr.shape == (small_corpus.height, 5, 32)  # ceil(10/2) layers
    assert meta["layer_stride"] == 2
    assert meta["prefix_only"] is True
    assert sidecar.height == small_corpus.height


# --------------------------------------------------------------------------- #
# train.py
# --------------------------------------------------------------------------- #


def test_quadruplet_id_split_keeps_siblings_together() -> None:
    qids = np.repeat(np.arange(20), 4)  # 20 quadruplets x 4 siblings
    train_idx, test_idx = quadruplet_id_split(qids, test_frac=0.25, seed=0)
    assert len(train_idx) + len(test_idx) == len(qids)
    assert set(train_idx).isdisjoint(set(test_idx))
    train_quads = set(qids[train_idx].tolist())
    test_quads = set(qids[test_idx].tolist())
    assert train_quads.isdisjoint(test_quads)


def test_train_probe_recovers_synthetic_label() -> None:
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    # signal: one direction encodes y; rest is noise.
    X = rng.normal(0, 0.5, size=(n, 32))
    X[:, 0] += 3.0 * (2 * y - 1)
    out = train_probe_at_layer(X[: n // 2], y[: n // 2], X[n // 2 :], y[n // 2 :], ProbeConfig())
    assert out["auroc"] > 0.95


def test_sweep_layers_finds_encoding_layer(tmp_path: Path, small_corpus) -> None:
    forward, set_labels = _make_synthetic_forward(
        label_layer=2, n_layers=6, hidden_dim=32, label_strength=8.0
    )
    labels = label_structured_vs_control(small_corpus)
    label_map = {
        build_prefix_prompt(row["sequence"]): int(row["role"] == "structured")
        for row in small_corpus.iter_rows(named=True)
    }
    set_labels(label_map)
    out = extract(
        stimuli_df=small_corpus,
        forward_fn=forward,
        config=ExtractionConfig(layer_stride=1, batch_size=8),
        output_dir=tmp_path,
        model_name="synth",
    )
    arr, sidecar, _ = load_activations(out["layers"], out["sidecar"], out["meta"])
    qids = sidecar["quadruplet_id"].to_numpy()
    sweep = sweep_layers(np.asarray(arr), labels, qids, ProbeConfig())
    # Layer 2 should have the highest AUROC, well above the noise layers.
    best = sweep.sort("auroc", descending=True).row(0, named=True)
    layer2_auroc = sweep.filter(pl.col("layer_index") == 2)["auroc"][0]
    assert best["layer_index"] == 2
    assert layer2_auroc > 0.9
    # Other layers should be near chance.
    other = sweep.filter(pl.col("layer_index") != 2)["auroc"].max()
    assert other < 0.75, f"non-encoding layer AUROC suspiciously high: {other}"


def test_levinstein_herrmann_transfer_holds(tmp_path: Path, small_corpus) -> None:
    forward, set_labels = _make_synthetic_forward(label_layer=1, n_layers=4, hidden_dim=32)
    labels = label_structured_vs_control(small_corpus)
    label_map = {
        build_prefix_prompt(row["sequence"]): int(row["role"] == "structured")
        for row in small_corpus.iter_rows(named=True)
    }
    set_labels(label_map)
    out = extract(
        stimuli_df=small_corpus,
        forward_fn=forward,
        config=ExtractionConfig(layer_stride=1, batch_size=8),
        output_dir=tmp_path,
        model_name="synth",
    )
    arr, sidecar, _ = load_activations(out["layers"], out["sidecar"], out["meta"])
    families = sidecar["family"].to_numpy()
    train_fams = ["periodic", "markov"]
    test_fams = ["hierarchical_pcfg", "cellular_automaton"]
    transfer = levinstein_herrmann_transfer(
        np.asarray(arr),
        families,
        labels,
        ProbeConfig(),
        train_families=train_fams,
        test_families=test_fams,
        layer=1,
    )
    # The synthetic label is family-independent (purely based on role),
    # so transfer AUROC should be high.
    assert transfer["auroc"] > 0.9


def test_cantos_compare_internal_when_gap_small() -> None:
    out = cantos_compare(auroc_with_answer=0.85, auroc_prefix_only=0.84, behavioral_accuracy=0.85)
    assert out["verdict"] == "internal"


def test_cantos_compare_leak_when_prefix_drops_and_tracks_behavior() -> None:
    out = cantos_compare(auroc_with_answer=0.85, auroc_prefix_only=0.55, behavioral_accuracy=0.86)
    assert out["verdict"] == "leak"
    assert out["answer_minus_prefix_gap"] > 0.05


def test_label_structured_vs_control_binary() -> None:
    sidecar = pl.DataFrame(
        {
            "quadruplet_id": [0, 0, 1, 1],
            "role": ["structured", "shuffled", "structured", "random_match"],
        }
    )
    labels = label_structured_vs_control(sidecar)
    assert labels.tolist() == [1, 0, 1, 0]
