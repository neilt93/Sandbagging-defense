"""Model backend abstraction for the pairwise preference harness.

Three backends:

  MockBackend       : deterministic synthetic preferences for unit tests
                      and pipeline validation without any model.
  HFBackend         : transformers, single-prompt forward-pass logits.
                      Works on CPU for tiny models; slow but portable.
  VLLMBackend       : vLLM SamplingParams with logprobs. Production path
                      for the Qwen2.5 0.5B-32B scaling sweep.

Each backend implements `score_pair(prompt, A_token, B_token) -> dict`,
returning the unnormalized logprobs of A_token and B_token at the next
position, plus enough metadata to reconstruct the call.

Backends are constructed by `get_backend(spec)` where spec is a dict with
a "kind" field. Heavyweight backends (HF, vLLM) lazy-import their deps so
the test suite can import the module without torch installed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class PairScore:
    logp_a: float
    logp_b: float
    extra: dict = field(default_factory=dict)

    @property
    def prob_a(self) -> float:
        m = max(self.logp_a, self.logp_b)
        ea = math.exp(self.logp_a - m)
        eb = math.exp(self.logp_b - m)
        return ea / (ea + eb)


class Backend(Protocol):
    name: str
    system_prompt: str | None

    def score_pair(self, prompt: str, a_token: str, b_token: str) -> PairScore: ...
    def score_completions(self, prompt: str, completions: list[str]) -> list[float]: ...
    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# MockBackend: synthetic deterministic preferences
# --------------------------------------------------------------------------- #


class MockBackend:
    """Returns scores driven by user-supplied scoring functions.

    Default `score_fn` prefers shorter LZ phrases (more structure).
    Default `mc_score_fn` reads the answer key off the prompt
    (`Answer: <letter>` token marker) when present, picking the correct
    completion. When `system_prompt` contains the substring 'sandbag',
    the mock systematically gives the WRONG answer on multiple-choice
    questions while leaving the structure preference intact — this lets
    the test suite verify the money plot's pattern (capability drops,
    preference holds) without a real model.
    """

    def __init__(
        self,
        name: str = "mock",
        score_fn: Callable[[str, str, str], tuple[float, float]] | None = None,
        mc_score_fn: Callable[[str, list[str], str | None], list[float]] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.name = name
        self.score_fn = score_fn or _default_mock_score_fn
        self.mc_score_fn = mc_score_fn or _default_mock_mc_score_fn
        self.system_prompt = system_prompt

    def score_pair(self, prompt: str, a_token: str, b_token: str) -> PairScore:
        logp_a, logp_b = self.score_fn(prompt, a_token, b_token)
        return PairScore(logp_a=logp_a, logp_b=logp_b, extra={"backend": "mock"})

    def score_completions(self, prompt: str, completions: list[str]) -> list[float]:
        return self.mc_score_fn(prompt, completions, self.system_prompt)

    def close(self) -> None:
        pass


def _default_mock_score_fn(prompt: str, a_token: str, b_token: str) -> tuple[float, float]:
    """Heuristic: prefer the side with the lower LZ76 phrase count.

    LZ76 is a stable structure proxy on length-128 sequences. We extract
    each side's sequence by taking the first comma-separated digit run
    after the 'A:' and 'B:' labels, then score using the same LZ76
    parser used in stimulus complexity. The mock backend's preferences
    therefore correctly track the corpus's structural ordering.

    Returns unnormalized log-odds.
    """
    import re

    a_match = re.search(r"A:\s*([0-9 ,]+)", prompt)
    b_match = re.search(r"B:\s*([0-9 ,]+)", prompt)
    if not a_match or not b_match:
        return 0.0, 0.0
    a_seq = [int(x) for x in re.findall(r"\d+", a_match.group(1))]
    b_seq = [int(x) for x in re.findall(r"\d+", b_match.group(1))]

    # Local import to avoid a cycle at module load time.
    from sandbagging_defense.stimuli.complexity import lz76_phrase_count

    a_lz = lz76_phrase_count(a_seq)
    b_lz = lz76_phrase_count(b_seq)
    # Convert to log-odds: lower LZ -> higher score. Scale so that a
    # 1-phrase gap maps to a meaningful logit difference.
    a_score = -a_lz / 8.0
    b_score = -b_lz / 8.0
    return a_score, b_score


def _default_mock_mc_score_fn(
    prompt: str, completions: list[str], system_prompt: str | None
) -> list[float]:
    """Multiple-choice scorer for MockBackend.

    Looks for an `Answer: <letter>` marker in the prompt indicating the
    ground-truth answer. When found:
      - normal mode (no 'sandbag' in system_prompt): give the correct
        completion the highest logit
      - sandbag mode ('sandbag' in system_prompt): give a deterministic
        WRONG answer (rotate to the next index) — simulates the
        capability-drop pattern that Phase 3 tests for.

    When no answer marker is present, returns equal logits.
    """
    import re

    n = len(completions)
    base = [0.0] * n
    match = re.search(r"Answer:\s*([A-Z])", prompt)
    if not match:
        return base
    correct_letter = match.group(1)
    correct_idx = ord(correct_letter) - ord("A")
    if not 0 <= correct_idx < n:
        return base
    is_sandbag = system_prompt is not None and "sandbag" in system_prompt.lower()
    target_idx = (correct_idx + 1) % n if is_sandbag else correct_idx
    base[target_idx] = 5.0  # large logit gap so argmax picks it
    return base


# --------------------------------------------------------------------------- #
# HFBackend: transformers single-prompt forward
# --------------------------------------------------------------------------- #


class HFBackend:
    """Single-prompt logits via a Hugging Face causal LM.

    Loads model lazily on first use. Single-call latency dominates so this
    backend is only useful for testing the pipeline with small models on
    CPU; the real pilot uses VLLMBackend.
    """

    def __init__(
        self,
        name: str,
        hf_id: str,
        device: str = "auto",
        dtype: str = "auto",
        system_prompt: str | None = None,
    ) -> None:
        self.name = name
        self.hf_id = hf_id
        self.device = device
        self.dtype = dtype
        self.system_prompt = system_prompt
        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Heavy imports are deferred so non-ml environments can still
        # import this module.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = (
            torch.float32
            if self.dtype == "fp32"
            else (torch.float16 if self.dtype == "fp16" else "auto")
        )
        device_map = (
            self.device
            if self.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, torch_dtype=torch_dtype, device_map=device_map
        )
        self._model.eval()

    def score_pair(self, prompt: str, a_token: str, b_token: str) -> PairScore:
        logps = self.score_completions(prompt, [a_token, b_token])
        return PairScore(
            logp_a=logps[0], logp_b=logps[1], extra={"backend": "hf", "hf_id": self.hf_id}
        )

    def score_completions(self, prompt: str, completions: list[str]) -> list[float]:
        self._ensure_loaded()
        import torch

        full = (
            self._build_chat(prompt) if hasattr(self._tokenizer, "apply_chat_template") else prompt
        )
        ids = self._tokenizer(full, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model(**ids)
        logp = out.logits[0, -1, :].float().log_softmax(dim=-1)
        results: list[float] = []
        for token in completions:
            tid = self._tokenizer.encode(token, add_special_tokens=False)[-1]
            results.append(float(logp[tid]))
        return results

    def _build_chat(self, user: str) -> str:
        msgs: list[dict[str, str]] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": user})
        return self._tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def close(self) -> None:
        self._model = None
        self._tokenizer = None


# --------------------------------------------------------------------------- #
# VLLMBackend: production path
# --------------------------------------------------------------------------- #


class VLLMBackend:
    """vLLM-backed pairwise scoring with logprobs sampling.

    Uses prompt logprobs at the post-prompt position to read off the A/B
    distribution without actually generating tokens. Lazy import of vllm
    keeps the module importable without CUDA.
    """

    def __init__(
        self,
        name: str,
        hf_id: str,
        system_prompt: str | None = None,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "auto",
    ) -> None:
        self.name = name
        self.hf_id = hf_id
        self.system_prompt = system_prompt
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self._llm: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM  # type: ignore[import-untyped]

        self._llm = LLM(
            model=self.hf_id,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype=self.dtype,
            enforce_eager=False,
        )
        self._tokenizer = self._llm.get_tokenizer()

    def score_pair(self, prompt: str, a_token: str, b_token: str) -> PairScore:
        logps = self.score_completions(prompt, [a_token, b_token])
        return PairScore(
            logp_a=logps[0], logp_b=logps[1], extra={"backend": "vllm", "hf_id": self.hf_id}
        )

    def score_completions(self, prompt: str, completions: list[str]) -> list[float]:
        self._ensure_loaded()
        from vllm import SamplingParams  # type: ignore[import-untyped]

        full = (
            self._build_chat(prompt) if hasattr(self._tokenizer, "apply_chat_template") else prompt
        )
        params = SamplingParams(
            temperature=0.0, max_tokens=1, logprobs=max(20, 4 * len(completions))
        )
        out = self._llm.generate([full], params, use_tqdm=False)
        first = out[0].outputs[0]
        lp_dict = first.logprobs[0]
        results: list[float] = []
        for token in completions:
            tid = self._tokenizer.encode(token, add_special_tokens=False)[-1]
            entry = lp_dict.get(tid)
            results.append(entry.logprob if entry is not None else float("-inf"))
        return results

    def _build_chat(self, user: str) -> str:
        msgs: list[dict[str, str]] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": user})
        return self._tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def close(self) -> None:
        self._llm = None
        self._tokenizer = None


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def get_backend(spec: dict) -> Backend:
    """Construct a backend from a dict spec.

    Dispatch on `kind`:
      {"kind": "mock", "name": "mock"}
      {"kind": "hf", "name": "...", "hf_id": "..."}
      {"kind": "vllm", "name": "...", "hf_id": "..."}
    """
    kind = spec["kind"]
    if kind == "mock":
        return MockBackend(name=spec.get("name", "mock"), system_prompt=spec.get("system_prompt"))
    if kind == "hf":
        return HFBackend(
            name=spec["name"],
            hf_id=spec["hf_id"],
            device=spec.get("device", "auto"),
            dtype=spec.get("dtype", "auto"),
            system_prompt=spec.get("system_prompt"),
        )
    if kind == "vllm":
        return VLLMBackend(
            name=spec["name"],
            hf_id=spec["hf_id"],
            system_prompt=spec.get("system_prompt"),
            gpu_memory_utilization=spec.get("gpu_memory_utilization", 0.85),
            dtype=spec.get("dtype", "auto"),
        )
    raise ValueError(f"unknown backend kind: {kind}")
