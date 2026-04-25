# Plan: Structure-Preference Probing as a Sandbagging-Resistant Capability Eval

This file is the canonical plan. The README points here. Decision gates
are tracked in `results/results-summary.md`.

## Thesis

Capability evaluations are vulnerable to sandbagging because the target
dimension is legible to the model: it can identify what to suppress.
We propose **evaluation-axis illegibility** as a defense. Probes where
the model may know it is being evaluated but cannot infer which latent
structural dimension is being measured should be harder to selectively
sandbag than benchmarks like MMLU. We instantiate this principle through
preference probes over stimuli with controlled structural complexity
(synthetic Kolmogorov-stratified sequences and stripped symbolic music),
and test whether behavioral and activation-level structure sensitivity
persists under instructed sandbagging conditions where standard benchmark
performance collapses.

## Goal

Workshop-quality empirical paper with one decisive figure: under
instructed sandbagging, MMLU and HumanEval scores drop substantially
while preference-probe accuracy on structured stimuli remains stable,
with activation probes recovering the structural axis even when
behavioral output flattens.

Target venue: NeurIPS 2026 workshop (SoLaR or SafeGenAI) as v1, with a
path to ICLR or SaTML 2026 main track if the activation arm is clean.

## Success criteria

The paper lands if at least three of the four hold:

1. Behavioral preference accuracy correlates with model scale on
   structured stimuli with perplexity controls.
2. The preference signal survives instructed sandbagging while MMLU
   drops by 20% or more.
3. Linear probes on mid-layer activations recover the structural axis
   with AUROC above 0.8 even under behavioral flattening.
4. The effect generalizes from synthetic sequences to stripped symbolic
   music.

If only (1) and (4) hold, it is a workshop paper. If (2) and (3) also
hold, it is a main-track paper.

## Failure modes to plan for

The negative result is "behavioral signal exists but disappears under
sandbagging in both behavior and activations," which would mean
instructed sandbagging penetrates the representation and the defense
fails. This is still publishable as a cautionary result at SoLaR, just
with different framing. The plan must keep the negative-result path
viable.

## Phases & decision gates

Each phase ends with a decision gate. Run one phase per session.
After each phase, update `results/results-summary.md` with the three or
four numbers that determine the next decision.

| Phase | Goal | Decision gate |
| --- | --- | --- |
| 0 | Scaffolding | smoke test passes |
| 1 | Stimulus generation + ground truth | LZ ordering test passes; within-quadruplet bigram surprisal gap < 5% |
| 2 | Behavioral pilot | preference > chance and scales with model size? proceed; flat? reduce scope; chance? try one more reframing round before quitting |
| 3 | Sandbagging conditions | preference holds while MMLU/HumanEval drop? proceed; preference drops too? pivot to negative-result framing |
| 4 | Activation probes | AUROC > 0.8 under sandbag with prefix-only and Cantos controls? strong paper |
| 5 | Music generalization | cross-domain probe transfer AUROC reported |
| 6 | Writeup | all four limitations (Cantos, Wataoka, Lost in Sequence, Levinstein-Herrmann) addressed in §8 |

## Working with Claude Code

- Run one phase per session. Each phase ends with a decision gate.
- Paste the relevant phase block as the initial prompt. Explicitly say
  "do not proceed past the tests for this phase, stop and report results."
- Keep `results/` gitignored; commit `results-summary.md` after each phase.
- Slow down at Phase 1 (stimulus correctness) and Phase 4 (probe leakage
  controls). Everything else is mechanical once those are right.
