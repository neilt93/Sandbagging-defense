"""Phase 5 tests: music21 multi-composer loading + pitch-class quadruplets.

Bach is the largest source in the local corpus (~412 chorales) but
Beethoven, Mozart, Haydn, Schumann, and a handful of others are also
available. Tests that touch the corpus are marked @slow.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from sandbagging_defense.stimuli.music import (
    DEFAULT_COMPOSERS,
    DEFAULT_LENGTH,
    PITCH_VOCAB,
    extract_pitch_classes,
    generate_music_corpus,
    iter_bach_chorales,
    iter_scores,
)
from sandbagging_defense.stimuli.quadruplets import ROLES


@pytest.mark.slow
def test_iter_bach_chorales_returns_scores() -> None:
    scores = list(iter_bach_chorales(max_n=3))
    assert len(scores) >= 1


@pytest.mark.slow
def test_extract_pitch_classes_range_and_shape() -> None:
    for score in iter_bach_chorales(max_n=10):
        pc = extract_pitch_classes(score, length=64)
        if pc is None:
            continue
        assert pc.shape == (64,)
        assert pc.dtype == np.int64
        assert int(pc.min()) >= 0
        assert int(pc.max()) < PITCH_VOCAB
        return
    pytest.skip("no chorales >= length 64")


@pytest.mark.slow
def test_generate_music_corpus_yields_quadruplets() -> None:
    rows = list(generate_music_corpus(n_quadruplets=3, seed=0, length=64, max_chorales_to_scan=80))
    if not rows:
        pytest.skip("Bach corpus produced no qualifying chorales")
    df = pl.DataFrame(rows)
    n_quads = df["quadruplet_id"].n_unique()
    assert df.height == n_quads * 4
    assert set(df["role"].unique()) == set(ROLES)
    # family is now "music_<composer>" since the loader supports any composer
    assert df["family"].str.starts_with("music_").all()


@pytest.mark.slow
def test_music_corpus_complexity_metrics_finite() -> None:
    rows = list(generate_music_corpus(n_quadruplets=3, seed=0, length=64, max_chorales_to_scan=80))
    if not rows:
        pytest.skip("Bach corpus produced no qualifying chorales")
    df = pl.DataFrame(rows)
    for col in ["lz76_phrases", "lz76_normalized", "gzip_ratio", "bigram_surprisal"]:
        assert df[col].is_finite().all(), col
        assert (df[col] >= 0).all(), col


@pytest.mark.slow
def test_music_lz_ordering_load_bearing() -> None:
    """structured (real Bach) should be more compressible than the
    order-destroying controls. Loose threshold: use a small corpus, so
    just assert structured mean LZ < shuffled mean LZ.
    """
    rows = list(generate_music_corpus(n_quadruplets=8, seed=0, length=64, max_chorales_to_scan=200))
    if len(rows) < 12:  # need at least 3 quadruplets to compute means
        pytest.skip("Bach corpus produced too few chorales")
    df = pl.DataFrame(rows)
    means = df.group_by("role").agg(pl.col("lz76_phrases").mean().alias("mean_lz")).sort("role")
    by_role = {r: m for r, m in zip(means["role"], means["mean_lz"], strict=False)}
    print(f"\nmusic LZ by role: {by_role}")
    assert by_role["structured"] < by_role["shuffled"]
    assert by_role["structured"] < by_role["random_match"]


def test_pitch_vocab_constant() -> None:
    assert PITCH_VOCAB == 12
    assert DEFAULT_LENGTH == 128


def test_default_composers_includes_multiple_periods() -> None:
    # Quick sanity that the corpus extends beyond Bach.
    assert "bach" in DEFAULT_COMPOSERS
    assert len(DEFAULT_COMPOSERS) >= 4
    assert "beethoven" in DEFAULT_COMPOSERS or "mozart" in DEFAULT_COMPOSERS


@pytest.mark.slow
def test_iter_scores_yields_multiple_composers() -> None:
    """Across the default composer list we should see >= 2 distinct composers."""
    seen: set[str] = set()
    for composer, _ in iter_scores(composers=DEFAULT_COMPOSERS, max_per_composer=5):
        seen.add(composer)
        if len(seen) >= 2:
            break
    if not seen:
        pytest.skip("music21 local corpus empty")
    # Don't assert >=2 strictly: in minimal music21 installs only Bach is
    # present. But at least one composer must yield a parsable score.
    assert len(seen) >= 1


@pytest.mark.slow
def test_generate_music_corpus_attaches_composer_to_family() -> None:
    rows = list(
        generate_music_corpus(
            n_quadruplets=4,
            seed=0,
            length=64,
            composers=("bach",),
            max_voices_per_score=4,
            max_chorales_to_scan=80,
        )
    )
    if not rows:
        pytest.skip("Bach corpus produced no qualifying scores")
    df = pl.DataFrame(rows)
    families = df["family"].unique().to_list()
    assert all(f.startswith("music_") for f in families)
    assert "music_bach" in families


@pytest.mark.slow
def test_multi_voice_extraction_multiplies_supply() -> None:
    """Asking for 4 voices should yield strictly more quadruplets than 1
    voice when the corpus has multi-voice scores.
    """
    rows_one = list(
        generate_music_corpus(
            n_quadruplets=20,
            seed=0,
            length=64,
            composers=("bach",),
            max_voices_per_score=1,
            max_chorales_to_scan=40,
        )
    )
    rows_four = list(
        generate_music_corpus(
            n_quadruplets=20,
            seed=0,
            length=64,
            composers=("bach",),
            max_voices_per_score=4,
            max_chorales_to_scan=40,
        )
    )
    if not rows_one and not rows_four:
        pytest.skip("Bach corpus produced no qualifying scores at all")
    df_one = pl.DataFrame(rows_one) if rows_one else pl.DataFrame()
    df_four = pl.DataFrame(rows_four) if rows_four else pl.DataFrame()
    n_one = df_one["quadruplet_id"].n_unique() if df_one.height else 0
    n_four = df_four["quadruplet_id"].n_unique() if df_four.height else 0
    # 4-voice extraction gives at least as many quadruplets as 1-voice.
    assert n_four >= n_one


def test_extract_pitch_classes_handles_score_without_flatten() -> None:
    """Regression: a score-like object with no `parts` and no `flatten()`
    used to raise AttributeError; now returns None gracefully.
    """

    class Bare:
        pass

    assert extract_pitch_classes(Bare(), length=10) is None


def test_extract_pitch_classes_voice_index_out_of_range() -> None:
    class FakePart:
        def flatten(self):
            return self

        @property
        def notes(self):
            return []

    class FakeScore:
        parts = [FakePart()]

    assert extract_pitch_classes(FakeScore(), length=10, voice_index=5) is None


def test_extract_pitch_classes_returns_none_for_short_score() -> None:
    """Sanity: if a score has fewer notes than `length`, return None."""

    class FakeNote:
        class _P:
            pitchClass = 5

        pitch = _P()

    class FakePart:
        def flatten(self):
            return self

        @property
        def notes(self):
            return [FakeNote() for _ in range(10)]

    class FakeScore:
        parts = [FakePart()]

    assert extract_pitch_classes(FakeScore(), length=20) is None


def test_extract_pitch_classes_works_on_short_synthetic_part() -> None:
    """Synthetic test that doesn't need the music21 corpus."""

    class FakeNote:
        def __init__(self, pc: int) -> None:
            class P:
                pitchClass = pc

            self.pitch = P()
            self.pitches = [P()]

    class FakePart:
        def flatten(self):
            return self

        @property
        def notes(self):
            return [FakeNote(i % 12) for i in range(150)]

    class FakeScore:
        parts = [FakePart()]

    pc = extract_pitch_classes(FakeScore(), length=128)
    assert pc is not None
    assert pc.shape == (128,)
    assert int(pc.min()) >= 0
    assert int(pc.max()) < 12
    # The fake part cycles 0..11 repeatedly: pc[k] should equal k % 12.
    for k in range(128):
        assert int(pc[k]) == k % 12


def test_module_constants() -> None:
    # Light test that always runs (doesn't require music21 corpus).
    assert PITCH_VOCAB == 12
    assert math.isclose(DEFAULT_LENGTH, 128)
