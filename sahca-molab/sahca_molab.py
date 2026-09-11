# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "minigrid>=2.3.1",
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
#     "pyyaml>=6.0",
#     "tqdm>=4.65.0",
#     "matplotlib>=3.7.0",
# ]
# ///

import marimo

__generated_with = "0.6.0"
app = marimo.App(width="medium")


@app.cell
def gpu_check():
    import marimo as mo
    import torch
    import platform

    gpu_available = torch.cuda.is_available()

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_info = mo.md(f"""
## ✅ GPU Detected

| Property | Value |
|----------|-------|
| **Device** | {gpu_name} |
| **VRAM** | {gpu_mem:.1f} GB |
| **CUDA** | {torch.version.cuda} |
| **PyTorch** | {torch.__version__} |
| **Platform** | {platform.system()} {platform.machine()} |
""")
    else:
        gpu_info = mo.md("""
## ⚠️ No GPU Detected

Running on CPU. Experiments will be ~10x slower.

**To attach GPU on molab:** Click the "notebook specs" button in the header.
""")

    device = "cuda" if gpu_available else "cpu"
    mo.output.replace(gpu_info)
    return device, gpu_available, gpu_info, mo, torch


@app.cell
def experiment_selector(mo):
    config_dropdown = mo.ui.dropdown(
        options={
            "DoorKey-6x6 Fixed": "configs/dk6_fixed.yaml",
            "DoorKey-6x6 Randomized (P1 CRITICAL)": "configs/dk6_randomized.yaml",
            "DoorKey-8x8 GPU (1M steps)": "configs/dk8_gpu.yaml",
            "Empty-8x8 (sanity check)": "configs/empty8.yaml",
        },
        value="DoorKey-6x6 Randomized (P1 CRITICAL)",
        label="Experiment Config",
    )

    method_dropdown = mo.ui.dropdown(
        options=["sahca", "grpo", "gigpo", "gagpo"],
        value="sahca",
        label="Credit Assignment Method",
    )

    seeds_slider = mo.ui.slider(
        start=1, stop=10, value=5, step=1,
        label="Number of Seeds",
    )

    mo.output.replace(
        mo.md("## Experiment Setup").center()
    )

    return config_dropdown, method_dropdown, seeds_slider


@app.cell
def show_selectors(mo, config_dropdown, method_dropdown, seeds_slider):
    mo.output.replace(
        mo.vstack([
            config_dropdown,
            method_dropdown,
            seeds_slider,
            mo.md(f"""
**Selected:** `{method_dropdown.value}` on `{config_dropdown.value}` with **{seeds_slider.value}** seeds
"""),
        ])
    )
    return


@app.cell
def run_experiment(mo):
    import subprocess
    import sys
    import os

    # Run P1 experiment directly (DoorKey-6x6 Randomized, SAHCA, 5 seeds)
    config_path = "configs/dk6_randomized.yaml"
    method = "sahca"
    seeds = 5

    cmd = [
        sys.executable, "-m", "experiments.run_multiseed",
        "--config", config_path,
        "--method", method,
        "--seeds", str(seeds),
        "--resume",
    ]

    mo.output.replace(
        mo.vstack([
            mo.md("## Running P1 Experiment (DoorKey-6x6 Randomized)"),
            mo.md(f"**Command:** `{' '.join(cmd)}`"),
            mo.md("Please wait..."),
        ])
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        output = result.stdout + "\n" + result.stderr
        status = "✅ Complete" if result.returncode == 0 else "❌ Failed"
        mo.output.replace(
            mo.vstack([
                mo.md(f"## {status}"),
                mo.md(f"```\n{output[-5000:]}\n```"),
            ])
        )
    except Exception as e:
        mo.output.replace(mo.md(f"## ❌ Error\n```\n{str(e)}\n```"))

    return


@app.cell
def results_viewer(mo):
    import json
    from pathlib import Path
    import os

    log_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "logs"

    if not log_dir.exists():
        mo.output.replace(mo.md("*No results directory found yet.*"))
        return

    json_files = sorted(log_dir.glob("*.json"))
    if not json_files:
        mo.output.replace(mo.md("*No result files found in logs/. Run an experiment first.*"))
        return

    rows = []
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
            finals = [e for e in data.get("entries", []) if e.get("final", False)]
            if finals:
                final = finals[-1]
                rows.append({
                    "File": jf.name,
                    "Method": data.get("method", "?"),
                    "Seed": data.get("seed", "?"),
                    "Success Rate": f"{final['success_rate']:.2f}",
                    "Mean Return": f"{final['mean_return']:.3f}",
                    "Steps/sec": f"{final.get('steps_per_sec', 0):.0f}",
                    "Time (s)": f"{final.get('elapsed_sec', 0):.0f}",
                })
        except Exception:
            continue

    if rows:
        mo.output.replace(
            mo.vstack([
                mo.md("## 📊 Results"),
                mo.ui.table(rows),
            ])
        )
    else:
        mo.output.replace(mo.md("*Result files found but no completed experiments.*"))

    return


@app.cell
def plot_viewer(mo):
    import subprocess
    import sys
    import os
    from pathlib import Path

    plot_button = mo.ui.run_button(label="📈 Generate Plots")
    mo.output.replace(plot_button)
    return plot_button


@app.cell
def generate_and_show_plots(mo, plot_button, subprocess, sys, os):
    from pathlib import Path

    if not plot_button.value:
        mo.output.replace(mo.md("*Click the Generate Plots button to create visualizations.*"))
        return

    project_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, "-m", "experiments.generate_plots"]

    subprocess.run(cmd, cwd=project_dir, capture_output=True)

    plot_dir = Path(project_dir) / "logs" / "plots"
    if not plot_dir.exists():
        mo.output.replace(mo.md("*No plots generated. Run experiments first.*"))
        return

    png_files = sorted(plot_dir.glob("*.png"))
    if not png_files:
        mo.output.replace(mo.md("*No plot files found.*"))
        return

    images = [mo.image(src=str(p), width=700) for p in png_files]
    mo.output.replace(
        mo.vstack([mo.md("## 📈 Plots")] + images)
    )

    return


@app.cell
def github_push(mo):
    import subprocess
    import os

    push_button = mo.ui.run_button(label="🔄 Git Push Results to GitHub")
    mo.output.replace(
        mo.vstack([
            mo.md("## 💾 Save Results"),
            mo.md("⚠️ **molab has no persistence!** Push to GitHub before your session ends."),
            push_button,
        ])
    )
    return push_button


@app.cell
def do_git_push(mo, push_button, subprocess, os):
    if not push_button.value:
        return

    project_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        # Stage results
        subprocess.run(["git", "add", "-f", "logs/*.json", "logs/plots/"], cwd=project_dir, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=project_dir, capture_output=True)

        # Commit
        result = subprocess.run(
            ["git", "commit", "-m", "Update experiment results from molab"],
            cwd=project_dir, capture_output=True, text=True,
        )

        # Push
        push_result = subprocess.run(
            ["git", "push"],
            cwd=project_dir, capture_output=True, text=True,
        )

        output = result.stdout + "\n" + push_result.stdout
        if push_result.returncode == 0:
            mo.output.replace(mo.md(f"## ✅ Pushed to GitHub\n```\n{output}\n```"))
        else:
            error = result.stderr + "\n" + push_result.stderr
            mo.output.replace(mo.md(f"## ⚠️ Push Issue\n```\n{output}\n{error}\n```"))
    except Exception as e:
        mo.output.replace(mo.md(f"## ❌ Error\n```\n{str(e)}\n```"))

    return


if __name__ == "__main__":
    app.run()
