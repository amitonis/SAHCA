"""Run SAHCA with different similarity thresholds.

Usage:
    python -m experiments.run_threshold --thresholds 0.7 0.8 0.9 0.95 --config configs/dk6_fixed.yaml --seeds 3
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="SAHCA threshold sensitivity sweep")
    parser.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.7, 0.8, 0.9, 0.95],
        help="Similarity thresholds to test",
    )
    parser.add_argument("--config", type=str, default="configs/dk6_fixed.yaml")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default="logs")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"Threshold sensitivity sweep")
    print(f"Thresholds: {args.thresholds}")
    print(f"Seeds per threshold: {args.seeds}")
    print(f"{'='*60}\n")

    all_results = {}

    for threshold in args.thresholds:
        print(f"\n--- Threshold: {threshold} ---")

        # Override threshold in config
        if "credit_assignment" not in config:
            config["credit_assignment"] = {}
        config["credit_assignment"]["similarity_threshold"] = threshold

        seed_results = []
        for seed in range(args.seeds):
            # Custom log dir per threshold
            log_dir = Path(args.log_dir) / f"threshold_{threshold}"
            log_dir.mkdir(parents=True, exist_ok=True)

            trainer = Trainer(
                config=config,
                method_name="sahca",
                seed=seed,
                log_dir=str(log_dir),
            )

            if args.max_steps:
                entries = trainer.train(max_steps=args.max_steps)
            else:
                entries = trainer.train()

            final = entries[-1]
            seed_results.append({
                "seed": seed,
                "success_rate": final["success_rate"],
                "mean_return": final["mean_return"],
            })

        all_results[threshold] = seed_results

    # Print summary
    import numpy as np

    print(f"\n{'='*60}")
    print(f"Threshold Sensitivity Summary")
    print(f"{'='*60}")
    print(f"{'Threshold':>10s}  {'Mean SR':>10s}  {'Std SR':>10s}  {'Mean Ret':>10s}")
    print(f"{'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}")

    for threshold, results in sorted(all_results.items()):
        srs = [r["success_rate"] for r in results]
        rets = [r["mean_return"] for r in results]
        print(f"{threshold:>10.2f}  {np.mean(srs):>10.3f}  {np.std(srs):>10.3f}  {np.mean(rets):>10.3f}")


if __name__ == "__main__":
    main()
