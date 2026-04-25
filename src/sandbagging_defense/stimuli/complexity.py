"""Per-stimulus complexity metrics: LZ76, gzip ratio, Shannon, surprisal.

LZ76 (Lempel-Ziv 1976) phrase parsing is the primary complexity readout —
it is well-defined on short strings, where gzip's framing overhead can
dominate. gzip ratio is reported as a sanity check.
"""

from __future__ import annotations

import gzip
import math
from collections import Counter

import numpy as np

LOG2 = math.log(2.0)


# --------------------------------------------------------------------------- #
# LZ76 phrase complexity
# --------------------------------------------------------------------------- #


def lz76_phrase_count(seq: np.ndarray | list[int]) -> int:
    """Count Lempel-Ziv 1976 distinct phrases in `seq`.

    Scan left-to-right; at each position extend the current phrase until
    either (a) the running substring has not appeared before in the prefix
    or (b) the end of seq is reached. Each terminated substring is one phrase.
    """
    if isinstance(seq, np.ndarray):
        seq = seq.tolist()
    n = len(seq)
    if n == 0:
        return 0
    seen: set[tuple[int, ...]] = set()
    phrases = 0
    i = 0
    while i < n:
        j = i + 1
        while j <= n and tuple(seq[i:j]) in seen:
            j += 1
        seen.add(tuple(seq[i:j]))
        phrases += 1
        i = j
    return phrases


def lz76_normalized(seq: np.ndarray | list[int]) -> float:
    """Asymptotic per-symbol complexity: c(n) * log2(n) / n."""
    n = len(seq)
    if n <= 1:
        return 0.0
    c = lz76_phrase_count(seq)
    return c * math.log2(n) / n


# --------------------------------------------------------------------------- #
# gzip ratio (sanity check)
# --------------------------------------------------------------------------- #


def gzip_ratio(seq: np.ndarray, replicate: int = 16) -> float:
    """gzip-compressed bytes / raw bytes after replicating `replicate` times.

    Replication amortizes gzip's framing overhead so the ratio reflects the
    intrinsic compressibility of the sequence, not the header constant.
    """
    arr = np.asarray(seq, dtype=np.uint8)
    raw = arr.tobytes() * replicate
    return len(gzip.compress(raw, compresslevel=9)) / len(raw)


# --------------------------------------------------------------------------- #
# Shannon entropy at orders 1, 2, 3
# --------------------------------------------------------------------------- #


def shannon_entropy(seq: np.ndarray | list[int], order: int = 1) -> float:
    """Empirical Shannon entropy of order-`order` blocks, in bits/block.

    For order 1 this is the marginal entropy. For order > 1 it is the joint
    entropy of length-`order` blocks divided by 1 (NOT per-symbol).
    """
    if isinstance(seq, np.ndarray):
        seq = seq.tolist()
    n = len(seq)
    if n < order or order < 1:
        return 0.0
    blocks = [tuple(seq[i : i + order]) for i in range(n - order + 1)]
    counts = Counter(blocks)
    total = sum(counts.values())
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return h


# --------------------------------------------------------------------------- #
# Surprisal under empirical unigram model fitted to the same sequence
# --------------------------------------------------------------------------- #


def unigram_surprisal(seq: np.ndarray | list[int], vocab: int) -> float:
    """Average -log2 p(token) under the empirical unigram of `seq` itself.

    With Laplace smoothing so a sequence missing a vocab token still scores.
    Within a quadruplet, all four siblings share unigram marginals by
    construction, so this should be near-equal across siblings.
    """
    if isinstance(seq, np.ndarray):
        seq = seq.tolist()
    n = len(seq)
    if n == 0:
        return 0.0
    counts = Counter(seq)
    smoothed_total = n + vocab
    surprisals = [-math.log2((counts[t] + 1) / smoothed_total) for t in seq]
    return float(np.mean(surprisals))


def bigram_surprisal(seq: np.ndarray | list[int], vocab: int) -> float:
    """Average -log2 p(token_t | token_{t-1}) under empirical bigram of `seq`."""
    if isinstance(seq, np.ndarray):
        seq = seq.tolist()
    n = len(seq)
    if n < 2:
        return 0.0
    bigrams = Counter(zip(seq[:-1], seq[1:], strict=False))
    unigrams = Counter(seq[:-1])
    surprisals = []
    for a, b in zip(seq[:-1], seq[1:], strict=False):
        # Laplace-smoothed conditional.
        num = bigrams[(a, b)] + 1
        den = unigrams[a] + vocab
        surprisals.append(-math.log2(num / den))
    return float(np.mean(surprisals))


# --------------------------------------------------------------------------- #
# Combined readout
# --------------------------------------------------------------------------- #


def compute_all_metrics(seq: np.ndarray, vocab: int) -> dict[str, float]:
    return {
        "lz76_phrases": lz76_phrase_count(seq),
        "lz76_normalized": lz76_normalized(seq),
        "gzip_ratio": gzip_ratio(seq),
        "shannon_h1": shannon_entropy(seq, order=1),
        "shannon_h2": shannon_entropy(seq, order=2),
        "shannon_h3": shannon_entropy(seq, order=3),
        "unigram_surprisal": unigram_surprisal(seq, vocab),
        "bigram_surprisal": bigram_surprisal(seq, vocab),
    }
