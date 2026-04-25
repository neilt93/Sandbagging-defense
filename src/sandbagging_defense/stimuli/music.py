"""Phase 5: symbolic-music quadruplet generation.

Loads scores from music21's local corpus and extracts pitch-class
sequences (mod 12). Supports multiple composers (Bach is the default
since the local corpus is heavy on chorales, but Beethoven, Mozart,
Haydn, and others are present too) and multiple voices per score so
the corpus is not bottlenecked by any single composer.

For each (composer, voice) source that yields a length-`length`
pitch-class sequence we build the same shuffled / markov_pastiche /
random_match controls as Phase 1.

Output schema mirrors Phase 1 so the same analysis modules work
without modification: one row per stimulus, four rows per quadruplet,
columns for sequence, family, role, and complexity metrics.

The vocab for music is 12 (pitch classes 0..11). Quadruplet controls
are vocab-agnostic, so the Phase 1 quadruplets module is reused
unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import polars as pl

from sandbagging_defense.stimuli.complexity import compute_all_metrics
from sandbagging_defense.stimuli.quadruplets import build_quadruplet

logger = logging.getLogger(__name__)

PITCH_VOCAB = 12
DEFAULT_LENGTH = 128


def extract_pitch_classes(
    score, length: int = DEFAULT_LENGTH, voice_index: int = 0
) -> np.ndarray | None:
    """Extract a length-`length` pitch-class sequence from one voice of `score`.

    `voice_index=0` selects the soprano (top voice) when multiple parts
    are present; higher indices select alto, tenor, bass, etc. Returns
    None if the chosen voice has fewer than `length` notes.
    """
    parts = list(score.parts) if hasattr(score, "parts") else []
    if parts:
        if voice_index >= len(parts):
            return None
        candidate = parts[voice_index]
    else:
        if voice_index != 0:
            return None
        candidate = score
    if not hasattr(candidate, "flatten"):
        return None
    notes = list(candidate.flatten().notes)
    if len(notes) < length:
        return None
    pitches: list[int] = []
    for note in notes[:length]:
        # Chord -> use the highest pitch (soprano-equivalent within the chord).
        if hasattr(note, "pitches") and len(note.pitches) > 0:
            pitches.append(int(max(p.pitchClass for p in note.pitches)))
        elif hasattr(note, "pitch"):
            pitches.append(int(note.pitch.pitchClass))
        else:
            return None
    return np.asarray(pitches, dtype=np.int64)


# Composers in the music21 local corpus we expect to find in non-trivial
# numbers. The list is conservative; missing composers are silently skipped.
DEFAULT_COMPOSERS: tuple[str, ...] = (
    "bach",
    "beethoven",
    "mozart",
    "haydn",
    "schumann",
    "chopin",
    "monteverdi",
    "joplin",
    "palestrina",
)


def iter_scores(
    composers: Iterable[str] = DEFAULT_COMPOSERS,
    file_extensions: tuple[str, ...] = ("xml", "krn"),
    max_per_composer: int | None = None,
):
    """Yield (composer, parsed_score) pairs from music21's local corpus.

    Iterates composers in order. Within each composer, scans up to
    `max_per_composer` files and skips entries that fail to parse.
    """
    from music21 import corpus

    for composer in composers:
        try:
            paths = corpus.search(composer, fileExtensions=list(file_extensions))
        except Exception as exc:  # noqa: BLE001
            logger.warning("corpus.search failed for composer=%s: %s", composer, exc)
            continue
        if max_per_composer is not None:
            paths = paths[:max_per_composer]
        for entry in paths:
            try:
                yield composer, entry.parse()
            except Exception as exc:  # noqa: BLE001
                logger.warning("failed to parse %s/%s: %s", composer, entry, exc)


# Backwards-compatible alias (Bach-only iteration; previously the only path).
def iter_bach_chorales(max_n: int | None = None):
    """Yield parsed scores from the Bach corpus only.

    Kept for the original Bach-specific call site; new code should use
    `iter_scores` to draw from the wider local corpus.
    """
    for _, score in iter_scores(composers=("bach",), max_per_composer=max_n):
        yield score


def generate_music_corpus(
    n_quadruplets: int,
    seed: int = 0,
    length: int = DEFAULT_LENGTH,
    vocab: int = PITCH_VOCAB,
    composers: Iterable[str] = DEFAULT_COMPOSERS,
    max_voices_per_score: int = 4,
    max_chorales_to_scan: int | None = None,
) -> Iterator[dict]:
    """Yield up to `n_quadruplets` quadruplets drawn from the local corpus.

    For each parsed score we try voices 0..min(parts, max_voices_per_score)
    and emit a quadruplet for every voice that gives a length-`length`
    pitch-class sequence. This multiplies the effective corpus by ~4x for
    chorale-style scores. Composer label and voice index are stored in
    the per-row params dict.

    Backward-compatible: the legacy `max_chorales_to_scan` argument
    remains; it caps the number of scores scanned across all composers.
    """
    parent_rng = np.random.default_rng(seed)
    produced = 0
    scan_budget = max_chorales_to_scan
    scanned = 0

    for composer, score in iter_scores(composers=list(composers)):
        if produced >= n_quadruplets:
            break
        if scan_budget is not None and scanned >= scan_budget:
            break
        scanned += 1
        n_parts = len(list(score.parts)) if hasattr(score, "parts") else 1
        n_voices = min(n_parts, max_voices_per_score) if n_parts > 0 else 1
        for voice_index in range(n_voices):
            if produced >= n_quadruplets:
                break
            seq = extract_pitch_classes(score, length=length, voice_index=voice_index)
            if seq is None:
                continue
            q_rng = np.random.default_rng(parent_rng.integers(0, 2**63 - 1))
            siblings = build_quadruplet(q_rng, seq, vocab)
            meta = getattr(score, "metadata", None)
            title = meta.title if meta and meta.title else f"score_{produced}"
            for role, sibling_seq in siblings.items():
                metrics = compute_all_metrics(sibling_seq, vocab)
                yield {
                    "quadruplet_id": produced,
                    "role": role,
                    "family": f"music_{composer}",
                    "params": json.dumps(
                        {"title": title, "composer": composer, "voice_index": voice_index}
                    ),
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
    parser.add_argument(
        "--composers",
        nargs="+",
        default=list(DEFAULT_COMPOSERS),
        help="Composer keywords to search music21's local corpus for",
    )
    parser.add_argument(
        "--voices",
        type=int,
        default=4,
        help="Max voices to extract per multi-voice score (1 = soprano only)",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(
        generate_music_corpus(
            n_quadruplets=args.n,
            length=args.length,
            seed=args.seed,
            composers=args.composers,
            max_voices_per_score=args.voices,
        )
    )
    df = pl.DataFrame(rows)
    df.write_parquet(args.out)
    n_quads = df["quadruplet_id"].n_unique() if df.height > 0 else 0
    print(f"wrote {df.height} rows ({n_quads} quadruplets x 4) to {args.out}")
    if df.height > 0:
        print("composer/family counts:")
        print(df.group_by("family").agg(pl.len().alias("n")).sort("family"))


if __name__ == "__main__":
    main()
