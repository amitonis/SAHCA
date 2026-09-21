# SAHCA Project Plan

## Goal

Study whether confidence-weighted soft-anchor credit assignment can improve
learning and generalization in sparse-reward MiniGrid tasks.

The project must determine whether the method works experimentally; success is
not assumed in advance.

## Current implementation

All code is in `sahca_molab.py`:

- MiniGrid environment wrappers
- Dense reward option
- CNN policy and value network
- GRPO, GiGPO, GAGPO, and SAHCA
- PPO-style trainer
- GPU detection
- Interactive Marimo controls
- JSON result files
- PyTorch checkpoints
- Learning curves

## Execution order

### P0 — Pipeline smoke test

- Environment: DoorKey-6x6 fixed
- Method: SAHCA
- Steps: 1K–10K
- Group size: 8–16 episodes
- Evaluation: 5–10 episodes

Check that training runs, CUDA is active, metrics are saved, and gradients are
finite.

### P1 — Basic learning check

- Environment: Empty-8x8, then DoorKey-6x6 fixed
- Methods: GRPO and SAHCA
- Steps: 50K–100K
- Seeds: at least 3

The purpose is to confirm that the implementation can learn before making
claims about SAHCA.

### P2 — Randomized benchmark

- Environment: DoorKey-6x6 randomized
- Methods: GRPO, GiGPO, GAGPO, SAHCA
- Steps: 300K per run
- Seeds: 5 when compute allows

Report success rate, mean return, learning-curve AUC, steps to target success,
episode length, runtime, and variation across seeds.

### P3 — SAHCA ablations

Test whether the result comes from the intended mechanism:

- hard/exact anchors only
- soft anchors without confidence
- full confidence-weighted SAHCA
- different similarity thresholds
- different temperatures
- no trajectory/global fallback
- random or uninformative embeddings

### P4 — Harder environment

Test DoorKey-8x8 or another longer-horizon task only after P2 and P3 are stable.
The research question is whether performance degrades gracefully as task
complexity increases.

### P5 — Generalization

Train on one set of layout seeds and evaluate on held-out layout seeds. Report
training success, held-out success, and the generalization gap.

## Publication standard

A publishable paper needs more than a high final score. It should include:

- a precise hypothesis
- strong and fairly tuned baselines
- multiple random seeds
- ablations isolating the contribution
- held-out evaluation
- learning efficiency, not only final success
- confidence intervals or uncertainty estimates
- reproducible code, configurations, and saved results

Possible paper directions are:

1. An algorithm paper if SAHCA consistently improves credit assignment.
2. An ablation/analysis paper if the main contribution is understanding when
   soft anchors help or fail.
3. A benchmark paper if the project develops a controlled anchor-sparsity and
   layout-generalization evaluation suite.

## Rules

- Do not claim superiority before multi-seed experiments support it.
- Do not change the state representation between comparisons without treating
  it as an ablation.
- Keep training and evaluation seeds separate.
- Record the configuration, seed, device, code version, and runtime for every run.
- Stop expensive experiments when the smoke test or baseline check fails.

## Immediate next action

Run the self-contained notebook on Empty-8x8 for 1K–10K steps. Then run
DoorKey-6x6 fixed for 50K steps with GRPO and SAHCA using the same seed and
training budget.
