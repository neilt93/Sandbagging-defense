"""Phase 5: Bach chorale loader + stripped pitch-class quadruplet generation.

Loads chorales from music21's local Bach corpus, extracts the soprano
(top) voice's pitch-class sequence (mod 12), and builds the same
shuffled / markov_pastiche / random_match controls as Phase 1.

Output schema mirrors Phase 1 so the same analysis modules work without
modification: one row per stimulus, four rows per quadruplet, columns
for sequence, family, role, and complexity metrics.

The vocab for music is 12 (pitch classes 0..11). Quadruplet controls
are vocab-agnostic, so the Phase 1 quadruplets module is reused
unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl

from sandbagging_defense.stimuli.complexity import compute_all_metrics
from sandbagging_defense.stimuli.quadruplets import build_quadruplet

logger = logging.getLogger(__name__)

PITCH_VOCAB = 12
DEFAULT_LENGTH = 128


def extract_pitch_classes(chorale_score, length: int = DEFAULT_LENGTH) -> np.ndarray | None:
    """Extract a length-`length` pitch-class sequence from a chorale.

    Strategy: walk the soprano (top) voice's flat notes in order, take
    each note's `pitchClass` (0..11), and truncate or skip-if-too-short.
    Returns None if the chorale has fewer than `length` notes in any
    extractable voice.
    """
    parts = list(chorale_score.parts) if hasattr(chorale_score, "parts") else []
    candidate = parts[0] if parts else chorale_score
    notes = list(candidate.flatten().notes)
    if len(notes) < length:
        return None
    pitches: list[int] = []
    for note in notes[:length]:
        # Chord -> use the highest pitch (soprano-equivalent).
        if hasattr(note, "pitches") and len(note.pitches) > 0:
            pitches.append(int(max(p.pitchClass for p in note.pitches)))
        elif hasattr(note, "pitch"):
            pitches.append(int(note.pitch.pitchClass))
        else:
            return None
    return np.asarray(pitches, dtype=np.int64)


def iter_bach_chorales(max_n: int | None = None):
    """Yield parsed chorale scores from music21's Bach corpus."""
    from music21 import corpus

    paths = corpus.search("bach", fileExtensions="xml")
    if max_n is not None:
        paths = paths[:max_n]
    for entry in paths:
        try:
            yield entry.parse()
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to parse %s: %s", entry, exc)


def generate_music_corpus(
    n_quadruplets: int,
    seed: int = 0,
    length: int = DEFAULT_LENGTH,
    vocab: int = PITCH_VOCAB,
    max_chorales_to_scan: int | None = None,
) -> Iterator[dict]:
    """Yield rows for up to `n_quadruplets` chorale-derived quadruplets.

    Each chorale that produces a length-`length` pitch-class sequence
    becomes one quadruplet (structured + 3 controls).
    """
    parent_rng = np.random.default_rng(seed)
    produced = 0
    scan_budget = max_chorales_to_scan or (n_quadruplets * 4)

    for q_id, score in enumerate(iter_bach_chorales(max_n=scan_budget)):
        if produced >= n_quadruplets:
            break
        seq = extract_pitch_classes(score, length=length)
        if seq is None:
            continue
        q_rng = np.random.default_rng(parent_rng.integers(0, 2**63 - 1))
        siblings = build_quadruplet(q_rng, seq, vocab)
        meta = score.metadata
        title = meta.title if meta and meta.title else f"chorale_{q_id}"
        for role, sibling_seq in siblings.items():
            metrics = compute_all_metrics(sibling_seq, vocab)
            yield {
                "quadruplet_id": produced,
                "role": role,
                "family": "bach_chorale",
                "params": json.dumps({"title": title}),
                "sequence": sibling_seq.tolist(),
                **metrics,
            }
        produced += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(
        generate_music_corpus(
            n_quadruplets=args.n,
            length=args.length,
            seed=args.seed,
        )
    )
    df = pl.DataFrame(rows)
    df.write_parquet(args.out)
    n_quads = df["quadruplet_id"].n_unique() if df.height > 0 else 0
    print(f"wrote {df.height} rows ({n_quads} quadruplets x 4) to {args.out}")


if __name__ == "__main__":
    main()
