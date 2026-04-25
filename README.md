# Sandbagging Defense via Illegible Evaluation Axes

Structure-preference probing as a sandbagging-resistant capability evaluation.

## Thesis

Capability evaluations are vulnerable to sandbagging because the target
dimension is legible to the model: it can identify what to suppress. We
propose **evaluation-axis illegibility** as a defense. Probes where the
model may know it is being evaluated but cannot infer which latent
structural dimension is being measured should be harder to selectively
sandbag than benchmarks like MMLU.

## Positioning

| Axis                     | MMLU / HumanEval | RepE / Linear Probes | Trojan / Backdoor Evals | Inverse Scaling Probes | **Illegible-Objective Probes (ours)** |
| ------------------------ | ---------------- | -------------------- | ----------------------- | ---------------------- | ------------------------------------- |
| Prompt realism           | High             | Low                  | Medium                  | Medium                 | High                                  |
| Elicitation              | Behavioral       | Activation-only      | Behavioral              | Behavioral             | Behavioral + activation               |
| Activation readout       | No               | Yes                  | Sometimes               | No                     | Yes                                   |
| Metric objectivity       | Ground-truth     | Probe-relative       | Trigger-relative        | Ground-truth           | Ground-truth (Kolmogorov-stratified)  |
| **Objective illegibility** | Low (named)    | Low (named)          | Medium                  | Medium                 | **High (latent structural axis)**     |

## Phases

| Phase | Goal                                            | Make target |
| ----- | ----------------------------------------------- | ----------- |
| 0     | Project scaffolding                             | `make install` |
| 1     | Stratified stimulus generation + ground truth   | `make stimuli` |
| 2     | Behavioral pilot: pairwise preference harness   | `make pilot` |
| 3     | Sandbagging conditions: MMLU/HumanEval vs ours  | `make sandbag` |
| 4     | Activation probing with leakage controls        | `make probes` |
| 5     | Music generalization (Bach chorales, stripped)  | `make music` |
| 6     | Paper writeup                                   | (manual) |

Each phase ends with a decision gate. See `paper/plan.md` for the full plan
and `results/results-summary.md` for cross-phase findings.

## Quickstart

```bash
make install          # core + dev deps (no torch / vllm)
make install-ml       # adds torch, transformers, vllm
make stimuli          # generate Phase 1 artifacts
make test             # run unit tests
```

## Layout

```
src/sandbagging_defense/
  stimuli/        # Kolmogorov-stratified sequence generators + complexity metrics
  elicitation/    # vLLM pairwise preference harness
  sandbagging/    # Sandbag conditions + MMLU/HumanEval drivers
  probes/         # Activation extraction + linear probe training
  analysis/       # Aggregation, figures, statistics
configs/          # Hydra configs (pilot, sandbag, probes)
results/          # Generated artifacts (gitignored except summary)
notebooks/        # Diagnostics
paper/            # NeurIPS workshop manuscript + figures
tests/            # Stimulus correctness tests (load-bearing)
```

## License

MIT.
