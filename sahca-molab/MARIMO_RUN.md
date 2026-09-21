# Run SAHCA in Marimo

The only required project file is `sahca_molab.py`. It contains the environment,
policy, credit-assignment methods, trainer, controls, checkpoints, JSON output,
and plots. It does not import from `src/`, `methods/`, `configs/`, or
`experiments/`. The notebook mirrors `src/` + `methods/`: SAHCA-Micro default
(alpha=0, beta=1, gamma=0), true GAGPO, normalized obs, sparse held-out eval.

## Marimo / molab

1. Upload `sahca_molab.py` to Marimo.
2. Let Marimo install the inline dependencies.
3. Attach the RTX 6000 GPU.
4. Run the notebook.
5. Start with Empty-8x8, 10K steps, 20 eval episodes (P0 smoke).
6. Then DoorKey-6x6 fixed 50K, GRPO vs SAHCA-micro, same seed.
7. Only then go randomized + multi-seed.

Training may use dense shaping, but evaluation is always sparse MiniGrid
reward on held-out seeds. Check the console for loss/KL/grad_finite and
SAHCA coverage.

The notebook writes flat files beside itself:

- `sahca_<method>_seed<seed>_<timestamp>.json`
- `sahca_<method>_seed<seed>_<timestamp>.pt`

No project folders are required.

## Suggested progression

1. Empty-8x8, 10K steps, smoke check (gradients finite, metrics saved).
2. DoorKey 6x6 fixed, 50K steps, compare SAHCA-micro and GRPO, same seed.
3. DoorKey 6x6 randomized + SPARSE eval, compare all four methods.
4. P3 ablations via the SAHCA preset dropdown (micro_only, macro_only, etc).
5. Increase to 300K steps and use multiple seeds only after short runs are stable.
