"""Generate comparison plots from JSON result files.

Usage:
    python -m experiments.generate_plots --log_dir logs --output_dir logs/plots
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots from experiment results")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory with JSON results")
    parser.add_argument("--output_dir", type=str, default="logs/plots", help="Output directory for plots")
    return parser.parse_args()


def load_results(log_dir: str) -> dict:
    """Load all JSON result files and group by method+env."""
    results = defaultdict(list)
    log_path = Path(log_dir)

    for json_file in log_path.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            method = data.get("method", "unknown")
            env_name = data.get("config", {}).get("env", {}).get("name", "unknown")
            key = f"{method}_{env_name}"
            results[key].append(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: skipping {json_file}: {e}")

    return dict(results)


def plot_learning_curves(results: dict, output_dir: Path):
    """Plot success rate over training steps for each method."""
    # Group by environment
    env_groups = defaultdict(dict)
    for key, runs in results.items():
        method = runs[0]["method"]
        env_name = runs[0]["config"]["env"]["name"]
        env_groups[env_name][method] = runs

    colors = {
        "grpo": "#e74c3c",
        "gigpo": "#3498db",
        "gagpo": "#2ecc71",
        "sahca": "#9b59b6",
    }

    for env_name, methods_data in env_groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for method_name, runs in sorted(methods_data.items()):
            color = colors.get(method_name, "#95a5a6")

            # Collect learning curves across seeds
            all_steps = defaultdict(list)
            all_returns = defaultdict(list)

            for run in runs:
                for entry in run.get("entries", []):
                    step = entry["step"]
                    all_steps[step].append(entry["success_rate"])
                    all_returns[step].append(entry["mean_return"])

            if not all_steps:
                continue

            steps = sorted(all_steps.keys())
            mean_sr = [np.mean(all_steps[s]) for s in steps]
            std_sr = [np.std(all_steps[s]) for s in steps]
            mean_ret = [np.mean(all_returns[s]) for s in steps]
            std_ret = [np.std(all_returns[s]) for s in steps]

            # Success rate plot
            axes[0].plot(steps, mean_sr, label=method_name.upper(), color=color, linewidth=2)
            axes[0].fill_between(
                steps,
                np.array(mean_sr) - np.array(std_sr),
                np.array(mean_sr) + np.array(std_sr),
                alpha=0.2, color=color,
            )

            # Return plot
            axes[1].plot(steps, mean_ret, label=method_name.upper(), color=color, linewidth=2)
            axes[1].fill_between(
                steps,
                np.array(mean_ret) - np.array(std_ret),
                np.array(mean_ret) + np.array(std_ret),
                alpha=0.2, color=color,
            )

        axes[0].set_title(f"Success Rate - {env_name}", fontsize=13)
        axes[0].set_xlabel("Environment Steps")
        axes[0].set_ylabel("Success Rate")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-0.05, 1.05)

        axes[1].set_title(f"Mean Return - {env_name}", fontsize=13)
        axes[1].set_xlabel("Environment Steps")
        axes[1].set_ylabel("Mean Return")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        safe_name = env_name.replace("/", "_").replace("-", "_")
        fig.savefig(output_dir / f"learning_curves_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: learning_curves_{safe_name}.png")


def plot_final_comparison(results: dict, output_dir: Path):
    """Bar chart of final success rates across methods."""
    env_groups = defaultdict(dict)
    for key, runs in results.items():
        method = runs[0]["method"]
        env_name = runs[0]["config"]["env"]["name"]

        final_srs = []
        for run in runs:
            finals = [e for e in run.get("entries", []) if e.get("final", False)]
            if finals:
                final_srs.append(finals[-1]["success_rate"])

        if final_srs:
            env_groups[env_name][method] = {
                "mean": np.mean(final_srs),
                "std": np.std(final_srs),
            }

    colors = {
        "grpo": "#e74c3c",
        "gigpo": "#3498db",
        "gagpo": "#2ecc71",
        "sahca": "#9b59b6",
    }

    for env_name, methods_data in env_groups.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        method_names = sorted(methods_data.keys())
        x = np.arange(len(method_names))
        means = [methods_data[m]["mean"] for m in method_names]
        stds = [methods_data[m]["std"] for m in method_names]
        bar_colors = [colors.get(m, "#95a5a6") for m in method_names]

        bars = ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in method_names], fontsize=12)
        ax.set_ylabel("Final Success Rate", fontsize=12)
        ax.set_title(f"Final Performance - {env_name}", fontsize=13)
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, mean_val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{mean_val:.2f}", ha="center", fontsize=11, fontweight="bold",
            )

        plt.tight_layout()
        safe_name = env_name.replace("/", "_").replace("-", "_")
        fig.savefig(output_dir / f"final_comparison_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: final_comparison_{safe_name}.png")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.log_dir)

    if not results:
        print(f"No JSON result files found in {args.log_dir}/")
        return

    print(f"Found {sum(len(v) for v in results.values())} result files across {len(results)} method-env combos")

    plot_learning_curves(results, output_dir)
    plot_final_comparison(results, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
