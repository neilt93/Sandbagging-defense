"""Phase 4: linear probes on activation arrays with leakage controls.

Given activations of shape (N, n_layers, hidden_dim) and a sidecar
parquet with the structural label, we train per-layer logistic
regression probes that predict the structural family of the stimulus.

Critical controls (mirroring the plan):

  Quadruplet-aware split
    Train/test split is done by quadruplet_id, NOT row index. This
    prevents the probe from memorizing per-quadruplet noise patterns
    that are shared across the four siblings.

  Cantos uniform-output control
    The probe is also evaluated under prefix-only activations — i.e.,
    activations extracted without any answer token in the prompt. If
    AUROC under prefix-only matches AUROC under answer-token activations,
    the probe is reading internal structure, not output-derived features.
    The extract module already produces prefix-only activations by
    default; this module accepts an optional second activations array
    to compare AUROC across the two regimes.

  Levinstein-Herrmann transfer
    The probe is trained on a subset of structural families and tested
    on held-out families. High transfer AUROC indicates the probe is
    learning structural sensitivity, not family-specific surface cues.

CLI: `python -m sandbagging_defense.probes.train --config configs/probes.yaml`
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    C: float = 1.0
    max_iter: int = 1000


def quadruplet_id_split(
    quadruplet_ids: np.ndarray, test_frac: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) split BY quadruplet, not by row.

    All rows with the same quadruplet_id end up on the same side, so the
    probe can't exploit per-quadruplet leakage.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(quadruplet_ids)
    rng.shuffle(unique)
    n_test = max(1, int(len(unique) * test_frac))
    test_quads = set(unique[:n_test].tolist())
    test_idx = np.array(
        [i for i, q in enumerate(quadruplet_ids) if int(q) in test_quads], dtype=np.int64
    )
    train_idx = np.array(
        [i for i, q in enumerate(quadruplet_ids) if int(q) not in test_quads], dtype=np.int64
    )
    return train_idx, test_idx


def train_probe_at_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: ProbeConfig,
) -> dict:
    """Fit a binary or one-vs-rest logistic regression and report AUROC."""
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test = scaler.transform(X_test)
    clf = LogisticRegression(C=config.C, max_iter=config.max_iter)
    clf.fit(Xs_train, y_train)
    if len(np.unique(y_train)) == 2:
        score = clf.predict_proba(Xs_test)[:, 1]
        auroc = float(roc_auc_score(y_test, score))
    else:
        # multi-class: macro AUROC (one-vs-rest probabilities).
        score = clf.predict_proba(Xs_test)
        auroc = float(roc_auc_score(y_test, score, multi_class="ovr", average="macro"))
    return {"auroc": auroc, "n_train": int(len(y_train)), "n_test": int(len(y_test))}


def sweep_layers(
    activations: np.ndarray,
    labels: np.ndarray,
    quadruplet_ids: np.ndarray,
    config: ProbeConfig,
    test_frac: float = 0.2,
    seed: int = 0,
) -> pl.DataFrame:
    """Train one probe per layer, return AUROC table."""
    train_idx, test_idx = quadruplet_id_split(quadruplet_ids, test_frac=test_frac, seed=seed)
    rows = []
    n_layers = activations.shape[1]
    for layer in range(n_layers):
        X = activations[:, layer, :]
        result = train_probe_at_layer(
            X[train_idx], labels[train_idx], X[test_idx], labels[test_idx], config
        )
        rows.append({"layer_index": layer, **result})
    return pl.DataFrame(rows).sort("layer_index")


def levinstein_herrmann_transfer(
    activations: np.ndarray,
    families: np.ndarray,  # str labels
    binary_label: np.ndarray,
    config: ProbeConfig,
    train_families: Iterable[str],
    test_families: Iterable[str],
    layer: int,
) -> dict:
    """Train on one set of families, test on another.

    `binary_label` is the structural target (e.g. structured vs control).
    """
    train_mask = np.isin(families, list(train_families))
    test_mask = np.isin(families, list(test_families))
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return {"auroc": float("nan"), "note": "empty split"}
    X_train = activations[train_mask, layer, :]
    X_test = activations[test_mask, layer, :]
    return train_probe_at_layer(
        X_train, binary_label[train_mask], X_test, binary_label[test_mask], config
    )


def cantos_compare(
    auroc_with_answer: float,
    auroc_prefix_only: float,
    behavioral_accuracy: float,
    tolerance: float = 0.05,
) -> dict:
    """Cantos check: compare prefix-only AUROC to answer-conditioned AUROC.

    If `auroc_prefix_only` is much LESS than `auroc_with_answer` AND
    `auroc_with_answer` tracks `behavioral_accuracy` closely, the probe
    is reading the model's output rather than its internal
    representation. Returns a structured judgement.
    """
    answer_minus_prefix = auroc_with_answer - auroc_prefix_only
    tracks_behavior = abs(auroc_with_answer - behavioral_accuracy) < tolerance
    return {
        "auroc_with_answer": auroc_with_answer,
        "auroc_prefix_only": auroc_prefix_only,
        "answer_minus_prefix_gap": answer_minus_prefix,
        "behavioral_accuracy": behavioral_accuracy,
        "auroc_tracks_behavior": tracks_behavior,
        "verdict": (
            "leak" if answer_minus_prefix > 2 * tolerance and tracks_behavior else "internal"
        ),
    }


def load_activations(layers_path: Path, sidecar_path: Path, meta_path: Path) -> tuple:
    meta = json.loads(meta_path.read_text())
    arr = np.memmap(
        layers_path,
        dtype=np.float32,
        mode="r",
        shape=(meta["n"], meta["n_layers_kept"], meta["hidden_dim"]),
    )
    sidecar = pl.read_parquet(sidecar_path)
    return arr, sidecar, meta


def label_structured_vs_control(sidecar: pl.DataFrame) -> np.ndarray:
    """Binary label: 1 for the structured sibling, 0 otherwise."""
    return (sidecar["role"] == "structured").to_numpy().astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--activations-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = OmegaConf.load(args.config)
    config = ProbeConfig(C=cfg.training.C)

    arr, sidecar, _meta = load_activations(
        args.activations_dir / "layers.npy",
        args.activations_dir / "sidecar.parquet",
        args.activations_dir / "meta.json",
    )
    labels = label_structured_vs_control(sidecar)
    qids = sidecar["quadruplet_id"].to_numpy()

    sweep = sweep_layers(np.asarray(arr), labels, qids, config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sweep.write_parquet(args.out)
    print(json.dumps({"sweep_path": str(args.out)}))


if __name__ == "__main__":
    main()
