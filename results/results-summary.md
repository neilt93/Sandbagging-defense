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

## Phase 2 — Behavioral pilot (TBD)

Required numbers:
- Per-model rank correlation (preference ↔ LZ): r =
- Cross-reframing Spearman: ρ =
- Scaling slope (model size ↔ accuracy):

Decision gate (one of):
- [ ] above chance + scales with size → proceed to Phase 3
- [ ] above chance, flat in scale → proceed with reduced scope
- [ ] at chance → try one more reframing round, then quit

---

## Phase 3 — Sandbagging conditions (TBD)

Required numbers (per condition):
- MMLU relative perf:
- HumanEval relative perf:
- Preference accuracy relative perf:

Decision gate:
- [ ] preference holds while benchmarks drop ≥ 20% → proceed to Phase 4
- [ ] preference also drops → pivot to negative-result framing

---

## Phase 4 — Activation probes (TBD)

Required numbers:
- Best-layer AUROC, normal:
- Best-layer AUROC, sandbag_targeted:
- Best-layer AUROC, sandbag_general:
- Cantos uniform-output control AUROC:
- Levinstein-Herrmann transfer AUROC:

Decision gate:
- [ ] AUROC > 0.8 under sandbag with prefix-only + Cantos controls

---

## Phase 5 — Music generalization (TBD)

Required numbers:
- Pilot accuracy on music quadruplets:
- Sandbag-condition accuracy on music:
- Cross-domain probe transfer AUROC (synthetic → music):

---

## Phase 6 — Writeup (TBD)
