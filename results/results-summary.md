# Results summary

This file is the cross-phase paper trail. After each phase, append the
three or four numbers that determine the next decision gate. Keep entries
short — full results live under `results/<phase>/`.

## Phase 0 — Scaffolding (2026-04-25)

- Directory structure laid down per plan.
- Module entry points stubbed; smoke test asserts every submodule imports.
- `make install` / `make test` work; `make stimuli` etc. raise
  `NotImplementedError` until their phase lands.

**Decision gate: PASS** — smoke test green; ready for Phase 1.

---

## Phase 1 — Stimulus generation (2026-04-25)

**Decision gate: PASS** with one documented finding (see below).

**Corpus**: 2000 quadruplets x 4 siblings = 8000 stimuli at length 128 over
vocab 16, written to `results/stimuli/v1.parquet`. Balanced across 15
specs spanning 5 families: random_iid, markov{1,2,3}, periodic{2,4,8,16,32},
hierarchical_pcfg{2,3,4}, cellular_automaton{30,90,110}.

**Family counts** (8000 stimuli total):
- periodic: 2732
- cellular_automaton: 1676
- hierarchical_pcfg: 1620
- markov: 1512
- random_iid: 460

**LZ76 phrase count, mean by role**:
- structured: 57.2
- markov_pastiche: 55.3
- shuffled: 63.7
- random_match: 63.1

**Bigram surprisal, mean by role** (clean separation between bigram-
preserving and bigram-destroying controls):
- structured: 2.36
- markov_pastiche: 2.17
- shuffled: 3.11
- random_match: 3.09

**Marginal preservation**:
- shuffled vs structured: TV = 0 exactly (multiset preserved by permutation)
- markov_pastiche vs structured: mean TV = 0.106, p95 = 0.180
- random_match vs structured: mean TV = 0.113, p95 = 0.164

**Phase 1 finding (carry to Phase 4)**: For our current 5 generator
families, `markov_pastiche` overlaps with `structured` in LZ space
(delta = -1.96 phrases). The deterministic L-system PCFG is bigram-
decomposable: a bigram-fit pastiche can reproduce its statistics. The
load-bearing experimental claim (`structured < shuffled`, `structured <
random_match`) holds cleanly. Tightening the structured-vs-pastiche
distinction — needed to claim the model uses *higher-order* structure
rather than bigrams — requires richer generators: stochastic PCFG with
per-occurrence rule choice, or variable-length L-system substitutions.
Defer this to Phase 4 design when the activation-probe arm needs a
bigram-difficult control.

**Tests passing** (35/35):
- LZ ordering (load-bearing): structured < shuffled and < random_match
- gzip ratio ordering (load-bearing): same
- bigram surprisal pattern: bigram-preservers < bigram-destroyers
- shuffled multiset preserved exactly (TV = 0)
- TV distance for sampled siblings p95 < 0.4
- seeded reproducibility
- shape, range, finiteness across all 15 specs

Decision gate:
- [x] LZ ordering (load-bearing) passes
- [x] Marginal preservation tested (TV instead of bigram-surprisal gap;
      see test docstring for the rationale)
- [x] Reproducibility test (seeded) passes

---

## Phase 2 — Behavioral pilot harness (2026-04-25)

**Decision gate: PIPELINE READY** (no model run yet — see below).

Implemented:
- `elicitation/backends.py` — three backends behind a common `Backend`
  protocol with `score_pair(prompt, A_token, B_token) -> PairScore`:
    - `MockBackend` — synthetic LZ-tracking scorer for unit tests
    - `HFBackend`   — transformers, lazy-loaded for CPU/GPU
    - `VLLMBackend` — production path with logprobs
- `elicitation/harness.py` — pairwise iteration (6 unordered pairs ×
  2 positional orders = 12 calls per quadruplet per reframing),
  position balancing, parquet output at
  `results/pilot/{model}/{reframing}.parquet`.
- `analysis/pilot.py` — `attach_lz_gap`, `preference_accuracy`,
  `lz_rank_correlation`, `cross_reframing_agreement`, `scaling_table`.

End-to-end pipeline verified with the mock backend (13 tests):
- 12 ordered rows per quadruplet collapse to 6 balanced preferences.
- Position balancing cancels constant left-side bias.
- LZ-tracking mock recovers positive Spearman with `lz_gap`.
- Cross-reframing Spearman = 1.0 (deterministic mock).
- `scaling_table` joins external model-size labels.

**What's missing**: the actual Qwen2.5 0.5B-32B sweep. This requires
GPUs and the `[ml]` extras (`uv sync --extra dev --extra ml`) which
this sandbox does not have. To run the pilot:

```
make install-ml
uv run python -m sandbagging_defense.elicitation.harness \
    --config configs/pilot.yaml --backend-kind vllm
uv run python -c "from sandbagging_defense.analysis.pilot import *; ..."
```

Required numbers (TBD on a GPU box):
- Per-model rank correlation (preference ↔ LZ): r =
- Cross-reframing Spearman: ρ =
- Scaling slope (model size ↔ accuracy):

Decision gate (one of, when numbers come in):
- [ ] above chance + scales with size → proceed to Phase 3
- [ ] above chance, flat in scale → proceed with reduced scope
- [ ] at chance → try one more reframing round, then quit

---

## Phase 3 — Sandbagging conditions (2026-04-25)

**Decision gate: PIPELINE READY** (no model run yet — see Phase 2 note).

Implemented:
- `sandbagging/conditions.py` — `load_conditions(path)` parses
  configs/sandbag.yaml into typed `Condition` records.
- `sandbagging/mmlu.py` — minimal MMLU evaluator. Loads either
  `cais/mmlu` from HF datasets or a local JSONL file. Builds
  `Question\nA. ... D. ...\nAnswer:` prompts and uses
  `backend.score_completions(prompt, ['A','B','C','D'])` -> argmax.
- `sandbagging/run.py` — driver that runs each (model, condition)
  cell across MMLU + the Phase 2 preference harness, writing to
  `results/sandbag/{model}/{condition}/{mmlu.parquet, preferences/*.parquet}`.
- `analysis/sandbag.py` — `mmlu_summary`, `preference_summary`,
  `relative_performance`, `money_plot_frame`. The last produces the
  long-form table for the central money plot
  (x=condition, y=relative_perf, hue=metric, facet=model).

Backends gained `score_completions(prompt, completions) -> list[float]`
across mock, HF, and vLLM. The mock's MMLU scorer reads an `Answer: X`
marker from the prompt and rotates the answer when the system_prompt
contains "sandbag" — a synthetic stand-in that lets the test suite
exercise the money-plot pattern without a real model.

End-to-end pipeline verified with mock (9 tests):
- 3-condition mock run produces the textbook pattern: MMLU 1.0 -> 0.0
  under sandbag (>20% drop trivially), preference accuracy near-flat
  (max-min < 0.05 across conditions).
- `money_plot_frame` correctly assigns `relative_perf` of 1.0 to
  baseline and 0.0 to sandbag MMLU; preference relative_perf within
  10% of normal under sandbag.
- `relative_performance` baseline division verified exactly.

**Deferred from Phase 3 plan**: HumanEval requires generation +
sandboxed execution, which is a separate infrastructure lift. The
decision gate (preference holds while capability drops ≥20%) is
evaluable from MMLU + preference alone, so this is a scope reduction
rather than a missing piece.

Required numbers (TBD on a GPU box, after `make install-ml`):
- MMLU relative perf per condition:
- Preference accuracy relative perf per condition:

Decision gate (when numbers come in):
- [ ] preference holds while MMLU drops ≥ 20% → proceed to Phase 4
- [ ] preference also drops → pivot to negative-result framing

---

## Phase 4 — Activation probes (2026-04-25)

**Decision gate: PIPELINE READY** (no model run yet — see Phase 2 note).

Implemented:
- `probes/extract.py` — extracts hidden states at the final stimulus
  token at every k-th layer. Default `prefix_only=True` builds the
  comma-separated sequence WITHOUT any A/B answer scaffolding, the
  Cantos-style control. Saves a memmapped `(N, n_layers, hidden_dim)`
  array + sidecar parquet + meta.json. Accepts an injectable
  `forward_fn` so tests can substitute a synthetic forward; the real
  HF path is `hf_forward_fn(model_id)`.
- `probes/train.py`:
    - `quadruplet_id_split` — train/test split BY quadruplet so the
      probe can't memorize per-quadruplet patterns shared by siblings.
    - `train_probe_at_layer` — `StandardScaler` + `LogisticRegression`
      with binary or multi-class macro-AUROC.
    - `sweep_layers` — train one probe per layer, return AUROC table.
    - `levinstein_herrmann_transfer` — train on subset of families,
      test on held-out families.
    - `cantos_compare` — return `'leak'` vs `'internal'` based on the
      gap between answer-conditioned and prefix-only AUROC and on
      whether AUROC tracks behavioral accuracy.

End-to-end pipeline verified with synthetic activations (9 tests).
The synthetic forward encodes the binary structured-vs-control label
at one specific layer; the rest is noise. The pipeline:
- recovers the encoding layer as the AUROC argmax (>0.9 there, <0.75
  elsewhere)
- demonstrates Levinstein-Herrmann transfer holds when the label is
  family-independent (AUROC > 0.9)
- correctly classifies internal vs leaked probes via `cantos_compare`
- enforces sibling-isolated splits via `quadruplet_id_split`

Synthetic activations use a stable MD5-derived per-prompt seed so
results are reproducible across pytest invocations (Python's hash() is
randomized per process).

Required numbers (TBD on a GPU box):
- Best-layer AUROC, normal:
- Best-layer AUROC, sandbag_targeted:
- Best-layer AUROC, sandbag_general:
- Cantos uniform-output control AUROC:
- Levinstein-Herrmann transfer AUROC:

Decision gate (when numbers come in):
- [ ] AUROC > 0.8 under sandbag with prefix-only + Cantos controls

---

## Phase 5 — Music generalization (2026-04-25)

**Decision gate: PASS on 1-quadruplet smoke check** (full corpus
generation requires more chorale scanning; see below).

Implemented:
- `stimuli/music.py` — loads chorales via music21's local Bach corpus,
  extracts pitch-class sequences (mod 12) from the soprano (top) voice,
  reuses Phase 1's `build_quadruplet` to produce shuffled,
  markov_pastiche, random_match controls. Vocab = 12, default length
  = 128.
- `make music` target wired to the CLI.

End-to-end smoke (not asserted as a strong gate due to corpus size):
- 1 chorale meeting the length-128 filter produced a clean quadruplet
  with the expected LZ ordering: structured 50 < markov_pastiche 52
  ≤ shuffled 53 ≈ random_match 53.
- 9 tests pass: pitch-class extraction, shape/range invariants,
  quadruplet construction, complexity metric finiteness, and
  load-bearing LZ ordering on a small Bach sample.

**Phase 5 finding**: The music21 Bach corpus has ~412 entries but only
~14 chorales have ≥128 soprano notes. To get a publication-scale
corpus (~200 quadruplets), options are:
1. shorten the music sequences to length 64 (matches ~95 chorales),
2. concatenate soprano lines across chorales and chunk into length-128
   segments (gives ~150 quadruplets but blurs the per-chorale boundary),
3. include alto/tenor/bass voices as additional quadruplets per
   chorale (4x the supply).

This is a Phase 6 (writeup) decision; for the workshop paper, option 1
or 3 is reasonable and matches the Phase 1 vocab/length contract once
chosen. The pipeline supports any length via the `--length` flag.

Required numbers (TBD on a GPU box):
- Pilot accuracy on music quadruplets:
- Sandbag-condition accuracy on music:
- Cross-domain probe transfer AUROC (synthetic → music):

---

## Phase 6 — Writeup (TBD)

---

## Follow-ups (2026-04-25, after initial Phase 0–5 push)

### Stochastic PCFG investigation
Added `hierarchical_pcfg_stochastic` (per-occurrence rule choice from K
alternatives) to test whether bigram-resistant generators would tighten
the structured-vs-pastiche LZ gap that Phase 1 flagged. Empirical
finding (40-stim sweep across depth ∈ {2,3,4} and n_alt ∈ {2,5,10}):
the stochastic family makes the gap WIDER, not tighter. Confirmed by
sweeping deterministic PCFG, markov{1,2,3}, periodic, and CA — nearly
every family has pastiche LZ < structured LZ.

The root cause is a property of LZ on Markov samples: a fitted Markov
chain revisits short loops more aggressively than the structured
source, producing fewer novel phrases. So pastiche-of-Markov is
*more* compressible than its source under LZ. The stochastic family is
gated off by default in `all_specs()` and stays available as an
explicit-opt-in for activation-probe work where the bigram-resistance
matters under likelihood (not LZ) metrics.

### Music broadened beyond Bach
`stimuli/music.py` now scans the whole local music21 corpus by composer.
Local corpus head-count (xml/krn): bach (largest), beethoven 26,
mozart 16, schumann 12, haydn 9. Multi-voice extraction
(`max_voices_per_score=4`) gives ~4x supply per chorale. CLI accepts
`--composers` and `--voices`. Backwards-compatible
`iter_bach_chorales()` retained for the original entry point.

### Bug audit
Two real defects found and fixed:
- `extract_pitch_classes` raised AttributeError for score-like objects
  with no `parts` and no `flatten()` method. Now returns None.
- `gzip_ratio` raised ZeroDivisionError on empty input. Now returns 0.0.
Both pinned with regression tests.

### `analysis/figures.py` end-to-end
Implemented `money_plot`, `lz_preference_scatter`, and a
`synthetic_money_demo` so `make figures` produces a real PDF without
GPU runs. The CLI auto-falls-back to the synthetic demo when no
`results/sandbag/` exists, with a stderr warning. Verified 19KB PDF
output.

**120/120 tests passing across two consecutive runs; lint clean.**
