"""Phase 1: stimulus generators for five Kolmogorov-stratified families.

All generators produce length-N integer sequences over a V-token vocabulary.
Generators are deterministic given a numpy.random.Generator (rng).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from sandbagging_defense.stimuli.complexity import compute_all_metrics
from sandbagging_defense.stimuli.quadruplets import build_quadruplet

# --------------------------------------------------------------------------- #
# Family signatures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StimulusSpec:
    family: str
    params: dict


# --------------------------------------------------------------------------- #
# Family 1: random_iid
# --------------------------------------------------------------------------- #


def random_iid(rng: np.random.Generator, length: int, vocab: int) -> np.ndarray:
    return rng.integers(0, vocab, size=length, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Family 2: markov_k
# --------------------------------------------------------------------------- #


def markov(
    rng: np.random.Generator,
    length: int,
    vocab: int,
    order: int,
    alpha: float = 0.3,
) -> np.ndarray:
    """Order-k Markov chain over `vocab`, transitions ~ Dirichlet(alpha).

    Small alpha gives sparse rows, which produces strong sequential structure.
    """
    n_states = vocab**order
    transitions = rng.dirichlet(np.full(vocab, alpha), size=n_states)  # (n_states, vocab)
    seq = np.empty(length, dtype=np.int64)
    seq[:order] = rng.integers(0, vocab, size=order)
    cumprobs = np.cumsum(transitions, axis=1)
    rand = rng.random(length - order)
    for t in range(order, length):
        idx = 0
        for j in range(order):
            idx = idx * vocab + int(seq[t - order + j])
        seq[t] = int(np.searchsorted(cumprobs[idx], rand[t - order]))
    return seq


# --------------------------------------------------------------------------- #
# Family 3: periodic
# --------------------------------------------------------------------------- #


def periodic(rng: np.random.Generator, length: int, vocab: int, period: int) -> np.ndarray:
    base = rng.integers(0, vocab, size=period, dtype=np.int64)
    reps = (length + period - 1) // period
    return np.tile(base, reps)[:length]


# --------------------------------------------------------------------------- #
# Family 4: hierarchical_pcfg (deterministic L-system on the alphabet)
# --------------------------------------------------------------------------- #


def hierarchical_pcfg(
    rng: np.random.Generator,
    length: int,
    vocab: int,
    depth: int,
    branching: int = 4,
) -> np.ndarray:
    """L-system: each symbol s expands to a length-`branching` replacement.

    After `depth` rewrites, total length is axiom_len * branching**depth.
    Pad/truncate to `length`. The replacement table is rng-sampled, so each
    stimulus has a fresh grammar.
    """
    rules = rng.integers(0, vocab, size=(vocab, branching), dtype=np.int64)
    needed = length
    expansions = branching**depth
    axiom_len = max(1, (needed + expansions - 1) // expansions)
    seq = rng.integers(0, vocab, size=axiom_len, dtype=np.int64)
    for _ in range(depth):
        seq = rules[seq].reshape(-1)
    if seq.shape[0] < length:
        seq = np.tile(seq, (length + seq.shape[0] - 1) // seq.shape[0])
    return seq[:length]


# --------------------------------------------------------------------------- #
# Family 4b: hierarchical_pcfg_stochastic
# --------------------------------------------------------------------------- #


def hierarchical_pcfg_stochastic(
    rng: np.random.Generator,
    length: int,
    vocab: int,
    depth: int,
    branching: int = 4,
    n_alternatives: int = 3,
) -> np.ndarray:
    """Stochastic L-system with K alternative rules per symbol.

    Each symbol s has `n_alternatives` possible expansions of length
    `branching`. At each expansion call, a rule is sampled uniformly
    PER OCCURRENCE (not per symbol identity), so the same parent at
    different positions can produce different children. The resulting
    sequence has hierarchical structure that bigram-only resampling
    cannot reproduce — addressing the Phase 1 finding that the
    deterministic L-system was bigram-decomposable.
    """
    rules = rng.integers(0, vocab, size=(vocab, n_alternatives, branching), dtype=np.int64)
    expansions = branching**depth
    axiom_len = max(1, (length + expansions - 1) // expansions)
    seq = rng.integers(0, vocab, size=axiom_len, dtype=np.int64)
    for _ in range(depth):
        choices = rng.integers(0, n_alternatives, size=seq.shape[0])
        seq = rules[seq, choices].reshape(-1)
    if seq.shape[0] < length:
        seq = np.tile(seq, (length + seq.shape[0] - 1) // seq.shape[0])
    return seq[:length]


# --------------------------------------------------------------------------- #
# Family 5: cellular_automaton
# --------------------------------------------------------------------------- #


def cellular_automaton(
    rng: np.random.Generator,
    length: int,
    vocab: int,
    rule: int,
    bits_per_token: int = 4,
) -> np.ndarray:
    """Elementary 1D CA over `length` columns; pack `bits_per_token` rows per token.

    Default vocab=16 ↔ bits_per_token=4. The CA runs for `bits_per_token`
    timesteps starting from a random binary row, with periodic boundary
    conditions. Each token encodes the 4-bit vertical column above its index.
    """
    if 2**bits_per_token != vocab:
        raise ValueError(f"vocab={vocab} requires bits_per_token=log2(vocab); got {bits_per_token}")
    if not 0 <= rule < 256:
        raise ValueError(f"elementary CA rule must be in [0, 256); got {rule}")
    rule_bits = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.int64)
    row = rng.integers(0, 2, size=length, dtype=np.int64)
    columns = np.empty((bits_per_token, length), dtype=np.int64)
    columns[0] = row
    for t in range(1, bits_per_token):
        left = np.roll(row, 1)
        right = np.roll(row, -1)
        idx = (left << 2) | (row << 1) | right
        row = rule_bits[idx]
        columns[t] = row
    weights = (1 << np.arange(bits_per_token))[:, None]
    return (columns * weights).sum(axis=0).astype(np.int64)


# --------------------------------------------------------------------------- #
# Spec enumeration: weighted set the corpus draws from
# --------------------------------------------------------------------------- #


def all_specs(include_stochastic_pcfg: bool = False) -> list[StimulusSpec]:
    """Default spec set used by Phase 1.

    `hierarchical_pcfg_stochastic` is gated off by default. We added it
    to test whether per-occurrence rule choice would tighten the
    structured-vs-pastiche LZ gap (the Phase 1 finding), but empirically
    it makes the gap WIDER: pastiche-of-Markov has a built-in tendency
    to revisit short loops and ends up with LZ count *below* the source
    for nearly all our families. The stochastic family remains available
    as a genuinely-bigram-resistant control under bigram-LIKELIHOOD
    metrics (not LZ); enable it explicitly when that distinction matters.
    """
    specs: list[StimulusSpec] = [StimulusSpec("random_iid", {})]
    for k in (1, 2, 3):
        specs.append(StimulusSpec("markov", {"order": k}))
    for p in (2, 4, 8, 16, 32):
        specs.append(StimulusSpec("periodic", {"period": p}))
    for d in (2, 3, 4):
        specs.append(StimulusSpec("hierarchical_pcfg", {"depth": d}))
    if include_stochastic_pcfg:
        for d in (2, 3, 4):
            specs.append(StimulusSpec("hierarchical_pcfg_stochastic", {"depth": d}))
    for r in (30, 90, 110):
        specs.append(StimulusSpec("cellular_automaton", {"rule": r}))
    return specs


def realize(spec: StimulusSpec, rng: np.random.Generator, length: int, vocab: int) -> np.ndarray:
    if spec.family == "random_iid":
        return random_iid(rng, length, vocab)
    if spec.family == "markov":
        return markov(rng, length, vocab, **spec.params)
    if spec.family == "periodic":
        return periodic(rng, length, vocab, **spec.params)
    if spec.family == "hierarchical_pcfg":
        return hierarchical_pcfg(rng, length, vocab, **spec.params)
    if spec.family == "hierarchical_pcfg_stochastic":
        return hierarchical_pcfg_stochastic(rng, length, vocab, **spec.params)
    if spec.family == "cellular_automaton":
        return cellular_automaton(rng, length, vocab, **spec.params)
    raise ValueError(f"unknown family: {spec.family}")


# --------------------------------------------------------------------------- #
# CLI: generate a corpus of N quadruplets to parquet
# --------------------------------------------------------------------------- #


def generate_corpus(
    n_quadruplets: int,
    length: int,
    vocab: int,
    seed: int,
) -> Iterator[dict]:
    """Yield one row per stimulus (4 rows per quadruplet)."""
    specs = all_specs()
    parent_rng = np.random.default_rng(seed)
    spec_choices = parent_rng.integers(0, len(specs), size=n_quadruplets)

    for q_id in range(n_quadruplets):
        spec = specs[int(spec_choices[q_id])]
        # Each quadruplet gets its own rng so reproducibility is per-id.
        q_rng = np.random.default_rng(parent_rng.integers(0, 2**63 - 1))
        structured = realize(spec, q_rng, length, vocab)
        siblings = build_quadruplet(q_rng, structured, vocab)
        for role, seq in siblings.items():
            metrics = compute_all_metrics(seq, vocab)
            yield {
                "quadruplet_id": q_id,
                "role": role,
                "family": spec.family,
                "params": json.dumps(spec.params, sort_keys=True),
                "sequence": seq.tolist(),
                **metrics,
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(generate_corpus(args.n, args.length, args.vocab, args.seed))
    df = pl.DataFrame(rows)
    df.write_parquet(args.out)
    print(f"wrote {len(rows)} rows ({args.n} quadruplets x 4) to {args.out}")
    print("family counts:")
    print(df.group_by("family").agg(pl.len().alias("n")).sort("family"))


if __name__ == "__main__":
    main()
