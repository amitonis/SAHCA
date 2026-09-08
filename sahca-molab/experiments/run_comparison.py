"""Run comparison across multiple credit assignment methods.

Usage:
    python -m experiments.run_comparison --config configs/dk6_fixed.yaml --methods grpo gigpo gagpo sahca --seeds 3
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-method comparison")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--methods", nargs="+", default=["grpo", "gigpo", "gagpo", "sahca"],
        help="Methods to compare",
    )
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per method")
    parser.add_argument("--resume", action="store_true", help="Resume completed seeds")
    parser.add_argument("--max_steps", type=int, default=None, help="Override total_env_steps")
    parser.add_argument("--log_dir", type=str, default="logs", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"Multi-method comparison")
    print(f"Config: {args.config}")
    print(f"Methods: {args.methods}")
    print(f"Seeds per method: {args.seeds}")
    print(f"{'='*60}\n")

    for method in args.methods:
        print(f"\n{'='*40}")
        print(f"  Running: {method}")
        print(f"{'='*40}\n")

        cmd = [
            sys.executable, "-m", "experiments.run_multiseed",
            "--config", args.config,
            "--method", method,
            "--seeds", str(args.seeds),
            "--log_dir", args.log_dir,
        ]

        if args.resume:
            cmd.append("--resume")
        if args.max_steps:
            cmd.extend(["--max_steps", str(args.max_steps)])

        result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))

        if result.returncode != 0:
            print(f"[ERROR] {method} failed with return code {result.returncode}")

    print(f"\n{'='*60}")
    print(f"All methods complete. Results in: {args.log_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
