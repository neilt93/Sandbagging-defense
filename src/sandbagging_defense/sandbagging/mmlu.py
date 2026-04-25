"""Minimal MMLU evaluator.

Loads MMLU questions from either:
  - the HF datasets `cais/mmlu` config (real path; requires `datasets`), or
  - a local JSONL file with one question per line.

Each question is a dict with keys:
  question : str
  choices  : list[str] of length 4
  answer   : int in 0..3 (index of the correct choice)
  subject  : str (optional)

For each question we build a prompt (with the answer marker the
MockBackend keys off in test mode) and ask the backend to score the
A/B/C/D completion logits. Argmax over the four logprobs gives the
prediction; accuracy is the fraction correct.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from sandbagging_defense.elicitation.backends import Backend

logger = logging.getLogger(__name__)

LETTERS = ["A", "B", "C", "D"]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_hf_mmlu(n: int | None, seed: int) -> list[dict]:
    """Load MMLU from HF datasets. Lazy-imported."""
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("cais/mmlu", "all", split="test")
    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    rows: list[dict] = []
    for row in ds:
        rows.append(
            {
                "question": row["question"],
                "choices": row["choices"],
                "answer": int(row["answer"]),
                "subject": row.get("subject", ""),
            }
        )
    return rows


def build_mmlu_prompt(question: dict, include_answer_marker: bool = False) -> str:
    """Render a question into a prompt the backend can score.

    When `include_answer_marker` is True, the ground-truth letter is
    appended in the form `Answer: <letter>`. This is *only* used by the
    mock-test path so the MockBackend can simulate sandbagging behavior
    without a real model. The production path leaves it False.
    """
    body = [question["question"], ""]
    for letter, choice in zip(LETTERS, question["choices"], strict=False):
        body.append(f"{letter}. {choice}")
    body.append("")
    if include_answer_marker:
        body.append(f"Answer: {LETTERS[question['answer']]}")
    body.append("Answer:")
    return "\n".join(body)


def evaluate(
    backend: Backend,
    questions: Iterable[dict],
    include_answer_marker: bool = False,
) -> pl.DataFrame:
    """Score every question; return a DataFrame with per-question rows."""
    rows = []
    for i, q in enumerate(questions):
        prompt = build_mmlu_prompt(q, include_answer_marker=include_answer_marker)
        logps = backend.score_completions(prompt, LETTERS)
        pred = max(range(len(LETTERS)), key=lambda k: logps[k])
        rows.append(
            {
                "qid": i,
                "subject": q.get("subject", ""),
                "answer": int(q["answer"]),
                "prediction": pred,
                "correct": pred == int(q["answer"]),
                **{f"logp_{LETTERS[k]}": logps[k] for k in range(len(LETTERS))},
            }
        )
    return pl.DataFrame(rows)


def accuracy(df: pl.DataFrame) -> float:
    if df.height == 0:
        return 0.0
    return float(df["correct"].mean())
