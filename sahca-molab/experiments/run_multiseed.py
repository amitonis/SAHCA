"""Run an experiment with multiple random seeds.

Usage:
    python -m experiments.run_multiseed --config configs/dk6_fixed.yaml --method sahca --seeds 5
    python -m experiments.run_multiseed --config configs/dk6_randomized.yaml --method sahca --seeds 5 --resume
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-seed experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--method", type=str, required=True, choices=["grpo", "gigpo", "gagpo", "sahca"])
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--resume", action="store_true", help="Skip seeds with existing complete results")
    parser.add_argument("--max_steps", type=int, default=None, help="Override total_env_steps")
    parser.add_argument("--log_dir", type=str, default="logs", help="Output directory for JSON results")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_name = config.get("env", {}).get("name", "unknown")
    results_summary = []

    print(f"\n{'='*60}")
    print(f"Multi-seed experiment: {args.method} on {env_name}")
    print(f"Seeds: {args.seeds}, Config: {args.config}")
    print(f"{'='*60}\n")

    for seed in range(args.seeds):
        # Check for existing results if resuming
        log_filename = f"{env_name}_{args.method}_seed{seed}.json".replace("/", "_")
        log_path = log_dir / log_filename

        if args.resume and log_path.exists():
            try:
                with open(log_path) as f:
                    existing = json.load(f)
                # Check if it has a final entry
                if any(e.get("final", False) for e in existing.get("entries", [])):
                    final = [e for e in existing["entries"] if e.get("final", False)][-1]
                    print(f"[SKIP] Seed {seed} already complete: SR={final['success_rate']:.2f}")
                    results_summary.append({
                        "seed": seed,
                        "success_rate": final["success_rate"],
                        "mean_return": final["mean_return"],
                        "skipped": True,
                    })
                    continue
            except (json.JSONDecodeError, KeyError):
                pass  # Corrupted file, re-run

        print(f"\n--- Seed {seed} ---")
        trainer = Trainer(
            config=config,
            method_name=args.method,
            seed=seed,
            log_dir=str(log_dir),
        )

        if args.max_steps:
            entries = trainer.train(max_steps=args.max_steps)
        else:
            entries = trainer.train()

        final = entries[-1]
        results_summary.append({
            "seed": seed,
            "success_rate": final["success_rate"],
            "mean_return": final["mean_return"],
            "skipped": False,
        })

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Summary: {args.method} on {env_name}")
    print(f"{'='*60}")
    print(f"{'Seed':>6s}  {'SR':>8s}  {'Return':>10s}  {'Status':>8s}")
    print(f"{'-'*6:>6s}  {'-'*8:>8s}  {'-'*10:>10s}  {'-'*8:>8s}")

    for r in results_summary:
        status = "SKIP" if r.get("skipped") else "DONE"
        print(f"{r['seed']:>6d}  {r['success_rate']:>8.2f}  {r['mean_return']:>10.3f}  {status:>8s}")

    if results_summary:
        import numpy as np
        srs = [r["success_rate"] for r in results_summary]
        rets = [r["mean_return"] for r in results_summary]
        print(f"\nMean SR:  {np.mean(srs):.3f} ± {np.std(srs):.3f}")
        print(f"Mean Ret: {np.mean(rets):.3f} ± {np.std(rets):.3f}")


if __name__ == "__main__":
    main()
