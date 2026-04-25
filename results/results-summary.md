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

## Phase 1 — Stimulus generation (TBD)

Required numbers:
- LZ ratio ordering: `structured < markov_pastiche <= shuffled < random_iid`?
- Within-quadruplet bigram surprisal gap: < 5%?
- Stimulus count per family.

Decision gate:
- [ ] LZ ordering test passes
- [ ] Bigram surprisal gap < 5%
- [ ] Reproducibility test (seeded) passes

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
