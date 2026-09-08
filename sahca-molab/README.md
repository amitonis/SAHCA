# SAHCA-molab: Soft-Anchor Credit Assignment on GPU

Exact-state credit assignment methods (GiGPO, GAGPO) achieve strong performance on fixed-layout environments but collapse under procedural variability (-75% and -98% success rate drop respectively), confirming that their advantage depends on exact state repetition. We introduce SAHCA-Micro, which replaces exact-state hashing with cosine-similarity soft-anchor clustering of learned policy embeddings, achieving 100% success on DoorKey-6x6 fixed without requiring state repetition. Ablations reveal that the soft-anchor component alone is the effective mechanism, while trajectory-level and hindsight terms actively interfere when combined.

## Quick Start

### Local
```bash
pip install -r requirements.txt
python -m experiments.run_multiseed --config configs/dk6_fixed.yaml --method sahca --seeds 1
```

### molab (GPU)
1. Push this repo to GitHub
2. Open [molab.marimo.io](https://molab.marimo.io/)
3. Load `sahca_molab.py` from your GitHub URL
4. Click "notebook specs" → attach GPU
5. Run cells to execute experiments

## Repo Structure

```
sahca-molab/
├── sahca_molab.py          # Marimo notebook (molab UI)
├── requirements.txt        # Dependencies
├── README.md
├── src/
│   ├── __init__.py
│   ├── env.py              # MiniGrid wrapper (dense reward + fixed seed)
│   ├── policy.py           # CNN policy (obs_dim=151, embed() for SAHCA)
│   └── trainer.py          # PPO training loop + JSON logging
├── methods/
│   ├── __init__.py          # Method registry
│   ├── base.py              # CreditAssignmentMethod ABC
│   ├── grpo.py              # GRPO baseline (trajectory-level)
│   ├── gigpo.py             # GiGPO (exact-hash micro-advantage)
│   ├── gagpo.py             # GAGPO (TD-error + grouped value proxy)
│   └── sahca.py             # SAHCA-Micro (soft-anchor clustering)
├── experiments/
│   ├── run_multiseed.py     # N seeds for one config
│   ├── run_comparison.py    # Multi-method sweep
│   ├── run_threshold.py     # Similarity threshold sweep
│   └── generate_plots.py   # Consolidate JSONs → plots
├── configs/
│   ├── dk6_fixed.yaml       # DoorKey-6x6 fixed (seed=42)
│   ├── dk6_randomized.yaml  # DoorKey-6x6 randomized
│   ├── dk8_gpu.yaml         # DoorKey-8x8 (1M steps, GPU only)
│   └── empty8.yaml          # Empty-8x8 (sanity check)
├── logs/                    # JSON results (gitignored)
└── checkpoints/             # Model checkpoints (gitignored)
```

## Experiments (Priority Order)

| Priority | Experiment | Command |
|----------|-----------|---------|
| **P1** | SAHCA-Micro DK6 Randomized, 5 seeds | `python -m experiments.run_multiseed --config configs/dk6_randomized.yaml --method sahca --seeds 5` |
| **P2** | All methods DK6, 3 seeds each | `python -m experiments.run_comparison --config configs/dk6_fixed.yaml --seeds 3` |
| **P3** | Threshold sensitivity 0.7/0.8/0.9/0.95 | `python -m experiments.run_threshold --thresholds 0.7 0.8 0.9 0.95` |
| **P4** | DK8 with 1M steps | `python -m experiments.run_multiseed --config configs/dk8_gpu.yaml --method sahca --seeds 3` |

## CPU Results (Reference)

### DoorKey-6x6 (300K steps)

| Method | Fixed SR | Fixed Ret | Rand SR | Rand Ret |
|--------|----------|-----------|---------|----------|
| GRPO   | 0.14     | -0.714    | 0.12    | -0.826   |
| GiGPO  | 0.82     | 1.015     | 0.20    | -0.734   |
| GAGPO  | 1.00     | 2.156     | 0.02    | -1.315   |
| SAHCA default | 0.00 | -1.078  | 0.02    | -0.870   |

GiGPO: **-75%** fixed→rand. GAGPO: **-98%** fixed→rand.

### SAHCA Ablation (DoorKey-6x6 fixed, 300K steps)

| Config | α | β | γ | Success | Return |
|--------|---|---|---|---------|--------|
| **micro_only** | 0.0 | 1.0 | 0.0 | **1.00** | **2.423** |
| low_hindsight | 1.0 | 1.0 | 0.3 | 0.96 | 1.845 |
| default | 1.0 | 1.0 | 1.0 | 0.48 | -0.118 |
| no_hindsight | 1.0 | 1.0 | 0.0 | 0.24 | -0.481 |
| macro_only (=GRPO) | 1.0 | 0.0 | 0.0 | 0.06 | -0.856 |

## SAHCA-Micro Default Config

```yaml
credit_assignment:
  similarity_threshold: 0.9
  cross_trajectory_only: true
  alpha: 0.0   # no macro (hurts: contradicts micro signal)
  beta: 1.0    # pure soft-anchor
  gamma: 0.0   # no hindsight (hurts: noisy early training)
```

## molab Platform

- **GPU**: NVIDIA RTX Pro 6000 Blackwell, 96 GB VRAM, 125 TFLOPS
- **Session**: 12 hours max, 90 min idle timeout
- **Cost**: Free (reasonable usage)
- **Persistence**: NONE — push results to GitHub before session ends!
- **Speed**: ~8-12x faster than CPU (350 → 4000 steps/sec)
