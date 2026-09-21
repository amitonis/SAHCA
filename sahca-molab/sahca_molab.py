# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.6.0",
#     "minigrid>=2.3.1",
#     "gymnasium>=0.29.0",
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
#     "matplotlib>=3.7.0",
# ]
# ///

"""Flat, self-contained SAHCA experiment notebook for Marimo.

Upload this single file to Marimo. It intentionally does not import anything
from local project folders such as src/, methods/, or experiments/.
"""

import marimo

app = marimo.App(width="full")


@app.cell
def imports():
    import json
    import time
    from collections import defaultdict
    from pathlib import Path

    import gymnasium as gym
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from gymnasium import spaces
    from minigrid.wrappers import ImgObsWrapper
    from torch.distributions import Categorical

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_text = (
            f"**GPU:** `{torch.cuda.get_device_name(0)}`  |  "
            f"**VRAM:** `{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB`  |  "
            f"**CUDA:** `{torch.version.cuda}`"
        )
    else:
        device = torch.device("cpu")
        gpu_text = "**GPU:** not detected; running on CPU"

    return Categorical, F, ImgObsWrapper, Path, defaultdict, device, gpu_text, gym, json, mo, nn, np, plt, spaces, time, torch


@app.cell
def runtime_info(mo, gpu_text, torch):
    mo.md(
        "# SAHCA — Marimo GPU experiment runner\n\n"
        "All code is contained in this notebook. No local folders or Python modules are required.\n\n"
        + gpu_text
        + f"  \n**PyTorch:** `{torch.__version__}`"
    )


@app.cell
def gpu_monitor(mo, torch):
    import shutil as _shutil
    import subprocess as _subprocess

    if not torch.cuda.is_available():
        _gpu_status = "CUDA is not available in this runtime."
    else:
        _allocated = torch.cuda.memory_allocated() / 2**30
        _reserved = torch.cuda.memory_reserved() / 2**30
        _peak = torch.cuda.max_memory_allocated() / 2**30
        _total_vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        _lines = [
            f"**PyTorch memory allocated:** `{_allocated:.2f} GB`",
            f"**PyTorch memory reserved:** `{_reserved:.2f} GB`",
            f"**Peak allocated:** `{_peak:.2f} GB`",
            f"**Total VRAM:** `{_total_vram:.2f} GB`",
        ]
        if _shutil.which("nvidia-smi"):
            _query = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"]
            _raw = _subprocess.check_output(_query, text=True, timeout=5).strip()
            _utilization, _used, _total_smi, _temperature = [part.strip() for part in _raw.split(",")]
            _lines.extend([
                f"**GPU utilization:** `{_utilization}%`",
                f"**System VRAM used:** `{_used} MiB / {_total_smi} MiB`",
                f"**GPU temperature:** `{_temperature} °C`",
            ])
        else:
            _lines.append("`nvidia-smi` is unavailable; showing PyTorch memory only.")
        _gpu_status = "  \n".join(_lines)
    mo.md("## GPU monitor\n\n" + _gpu_status + "\n\n*Rerun this cell for a fresh reading.*")


@app.cell
def core(Categorical, F, ImgObsWrapper, Path, defaultdict, gym, json, nn, np, spaces, time, torch):
    # ----------------------------- Environment -----------------------------
    def state_hash(obs):
        import hashlib
        return hashlib.md5(np.asarray(obs).tobytes()).hexdigest()[:16]

    def manhattan(a, b):
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

    class FlatDirObs(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            # Normalized to [0,1]: raw 0-255 pixels destabilized the CNN.
            self.observation_space = spaces.Box(0.0, 1.0, (151,), dtype=np.float32)

        def observation(self, obs):
            image = np.asarray(obs, dtype=np.float32).reshape(-1) / 255.0
            direction = np.zeros(4, dtype=np.float32)
            direction[int(self.unwrapped.agent_dir)] = 1.0
            return np.concatenate([image, direction])

    class DenseReward(gym.Wrapper):
        def __init__(self, env, enabled=True):
            super().__init__(env)
            self.enabled = enabled

        def _goal(self):
            grid = self.unwrapped.grid
            for x in range(grid.width):
                for y in range(grid.height):
                    cell = grid.get(x, y)
                    if cell is not None and cell.type == "goal":
                        return (x, y)
            return None

        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            self.goal = self._goal()
            self.previous_distance = manhattan(self.unwrapped.agent_pos, self.goal) if self.goal else None
            self.has_key = False
            self.door_opened = False
            info["agent_pos"] = tuple(self.unwrapped.agent_pos)
            return obs, info

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            info["agent_pos"] = tuple(self.unwrapped.agent_pos)
            info["sparse_reward"] = float(reward)
            if not self.enabled:
                info["dense_reward"] = float(reward)
                return obs, reward, terminated, truncated, info

            shaped = -0.01
            carrying = self.unwrapped.carrying
            if not self.has_key and carrying is not None and carrying.type == "key":
                self.has_key = True
                shaped += 0.5

            if not self.door_opened:
                grid = self.unwrapped.grid
                for x in range(grid.width):
                    for y in range(grid.height):
                        cell = grid.get(x, y)
                        if cell is not None and cell.type == "door" and cell.is_open:
                            self.door_opened = True
                            shaped += 0.5
                            break
                    if self.door_opened:
                        break

            if self.goal is not None and self.previous_distance is not None:
                distance = manhattan(self.unwrapped.agent_pos, self.goal)
                shaped += 0.1 * (self.previous_distance - distance)
                self.previous_distance = distance
            if terminated:
                shaped += reward
            info["dense_reward"] = float(shaped)
            return obs, shaped, terminated, truncated, info

    def make_env(name, seed=None, dense_reward=True):
        env = gym.make(name)
        env = ImgObsWrapper(env)
        env = FlatDirObs(env)
        env = DenseReward(env, dense_reward)
        if seed is not None:
            class FixedSeed(gym.Wrapper):
                def reset(self, **kwargs):
                    kwargs["seed"] = seed
                    return super().reset(**kwargs)
            env = FixedSeed(env)
        return env

    # -------------------------------- Policy --------------------------------
    class Policy(nn.Module):
        def __init__(self, obs_dim=151, hidden_dim=256, action_dim=7):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 7 * 7 + 4, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.actor = nn.Linear(hidden_dim, action_dim)
            self.critic = nn.Linear(hidden_dim, 1)
            for layer in [self.conv1, self.conv2, self.fc1, self.fc2]:
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(self.actor.weight, gain=0.01)
            nn.init.zeros_(self.actor.bias)

        def features(self, obs):
            grid = obs[:, :147].reshape(-1, 7, 7, 3).permute(0, 3, 1, 2)
            direction = obs[:, 147:]
            x = F.relu(self.conv1(grid))
            x = F.relu(self.conv2(x)).reshape(obs.shape[0], -1)
            x = torch.cat([x, direction], dim=-1)
            return F.relu(self.fc2(F.relu(self.fc1(x))))

        def forward(self, obs):
            features = self.features(obs)
            return self.actor(features), self.critic(features)

        def sample(self, obs):
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)
            features = self.features(obs)
            distribution = Categorical(logits=self.actor(features))
            action = distribution.sample()
            return action.item(), distribution.log_prob(action).item(), self.critic(features).item(), features.detach().squeeze(0).cpu().numpy()

    # ------------------------- Credit assignment ---------------------------
    def zscore(values):
        values = np.asarray(values, dtype=np.float64)
        std = values.std()
        return np.zeros_like(values) if std < 1e-8 else (values - values.mean()) / std

    class CreditMethod:
        def compute(self, trajectories):
            raise NotImplementedError

        @staticmethod
        def macro(trajectories):
            values = zscore([sum(step["reward"] for step in traj) for traj in trajectories])
            return [[float(values[i])] * len(traj) for i, traj in enumerate(trajectories)]

    class GRPO(CreditMethod):
        def compute(self, trajectories):
            return self.macro(trajectories)

    class GiGPO(CreditMethod):
        def compute(self, trajectories):
            output = self.macro(trajectories)
            groups = defaultdict(list)
            for ti, traj in enumerate(trajectories):
                for si, step in enumerate(traj):
                    groups[step["state_hash"]].append((ti, si, step["return"]))
            for members in groups.values():
                if len(members) < 2:
                    continue
                values = zscore([member[2] for member in members])
                for (ti, si, _), value in zip(members, values):
                    output[ti][si] = float(value)
            return output

    class GAGPO(CreditMethod):
        """Exact-state grouping with TD-error + grouped value proxy.

        Value proxy = mean return of all steps sharing the hash.
        Advantage = z-scored (return - proxy). Falls back to GRPO.
        """
        def compute(self, trajectories):
            groups = defaultdict(list)
            for ti, traj in enumerate(trajectories):
                for si, step in enumerate(traj):
                    groups[step["state_hash"]].append((ti, si, step["return"]))
            proxies = {h: float(np.mean([m[2] for m in members])) for h, members in groups.items()}
            output = self.macro(trajectories)
            for h, members in groups.items():
                if len(members) < 2:
                    continue
                proxy = proxies[h]
                errors = np.asarray([m[2] - proxy for m in members], dtype=np.float64)
                std = errors.std()
                values = np.zeros_like(errors) if std < 1e-8 else (errors - errors.mean()) / std
                for (ti, si, _), value in zip(members, values):
                    output[ti][si] = float(value)
            return output

    class SAHCA(CreditMethod):
        """SAHCA-Micro: greedy cosine-similarity soft-anchor clustering.

        Validated default (alpha=0, beta=1, gamma=0) = pure soft-anchor.
        Ablations showed macro/hindsight terms interfere (see README table:
        micro_only 100% SR vs default-mix 48% vs macro_only 6%).
        """
        def __init__(self, threshold=0.9, alpha=0.0, beta=1.0, gamma=0.0, cross_trajectory_only=True):
            self.threshold = threshold
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.cross_trajectory_only = cross_trajectory_only
            self.last_stats = {}

        def compute(self, trajectories):
            n = len(trajectories)
            output = [[0.0] * len(traj) for traj in trajectories]
            if self.alpha > 0:
                macro = self.macro(trajectories)
                for i in range(n):
                    for j in range(len(trajectories[i])):
                        output[i][j] += self.alpha * macro[i][j]
            if self.beta > 0:
                micro = self._soft_anchor(trajectories)
                for i in range(n):
                    for j in range(len(trajectories[i])):
                        output[i][j] += self.beta * micro[i][j]
            if self.gamma > 0:
                hind = self._hindsight(trajectories)
                for i in range(n):
                    for j in range(len(trajectories[i])):
                        output[i][j] += self.gamma * hind[i][j]
            return output

        def _soft_anchor(self, trajectories):
            flat = [(ti, si, step) for ti, traj in enumerate(trajectories) for si, step in enumerate(traj)]
            micro = [[0.0] * len(traj) for traj in trajectories]
            if not flat:
                self.last_stats = {"soft_anchor_coverage": 0.0, "mean_neighbors": 0.0}
                return micro
            embeddings = np.asarray([item[2]["embedding"] for item in flat], dtype=np.float32)
            embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
            assigned = [False] * len(flat)
            clustered = 0
            for i in range(len(flat)):
                if assigned[i]:
                    continue
                sims = embeddings[i] @ embeddings.T
                cluster = [i]
                for j in range(i + 1, len(flat)):
                    if assigned[j]:
                        continue
                    if self.cross_trajectory_only and flat[j][0] == flat[i][0]:
                        continue
                    if sims[j] >= self.threshold:
                        cluster.append(j)
                if len(cluster) < 2:
                    continue
                for idx in cluster:
                    assigned[idx] = True
                returns = np.asarray([flat[k][2]["return"] for k in cluster], dtype=np.float64)
                std = returns.std()
                vals = np.zeros_like(returns) if std < 1e-8 else (returns - returns.mean()) / std
                for idx, val in zip(cluster, vals):
                    ti, si, _ = flat[idx]
                    micro[ti][si] = float(val)
                clustered += len(cluster)
            self.last_stats = {
                "soft_anchor_coverage": clustered / max(1, len(flat)),
                "mean_neighbors": float(clustered) / max(1, len(trajectories)),
            }
            return micro

        def _hindsight(self, trajectories):
            out = [[0.0] * len(traj) for traj in trajectories]
            for ti, traj in enumerate(trajectories):
                if not traj:
                    continue
                goal = traj[-1].get("agent_pos", (0, 0))
                dists = np.asarray([-abs(s.get("agent_pos", (0, 0))[0] - goal[0]) - abs(s.get("agent_pos", (0, 0))[1] - goal[1]) for s in traj], dtype=np.float64)
                std = dists.std()
                vals = np.zeros_like(dists) if std < 1e-8 else (dists - dists.mean()) / std
                for si, val in enumerate(vals):
                    out[ti][si] = float(val)
            return out

    METHODS = {"GRPO": GRPO, "GiGPO": GiGPO, "GAGPO": GAGPO, "SAHCA": SAHCA}

    # -------------------------------- Trainer -------------------------------
    class Trainer:
        def __init__(self, config, method_name, device, seed=0):
            self.config = config
            self.method_name = method_name
            self.device = device
            self.seed = seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            self._train_rng = np.random.default_rng(seed)
            self.eval_base_seed = 1_000_000 + seed
            self._eval_count = 0
            env_cfg, train_cfg = config["env"], config["training"]
            self.env_name = env_cfg["name"]
            self.fixed_seed = env_cfg.get("fixed_seed")
            self.dense_reward = env_cfg.get("dense_reward", True)
            self.group_size = int(train_cfg["group_size"])
            self.total_steps = int(train_cfg["total_env_steps"])
            self.eval_interval = int(train_cfg["eval_interval"])
            self.eval_episodes = int(train_cfg["eval_episodes"])
            self.policy = Policy(**config["policy"]).to(device)
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
            ca = config.get("credit_assignment", {})
            if method_name == "SAHCA":
                self.method = SAHCA(ca.get("similarity_threshold", 0.9), ca.get("alpha", 0.0), ca.get("beta", 1.0), ca.get("gamma", 0.0), ca.get("cross_trajectory_only", True))
            else:
                self.method = METHODS[method_name]()
            self.entries, self.steps = [], 0
            self.last_update = {}

        def collect(self):
            trajectories = []
            self.policy.eval()
            for _ in range(self.group_size):
                episode_seed = self.fixed_seed if self.fixed_seed is not None else int(self._train_rng.integers(0, 2**31 - 1))
                env = make_env(self.env_name, episode_seed, self.dense_reward)
                obs, _ = env.reset()
                trajectory, done = [], False
                while not done:
                    action, log_prob, value, embedding = self.policy.sample(torch.tensor(obs, dtype=torch.float32, device=self.device))
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    trajectory.append({"obs": np.asarray(obs), "action": action, "log_prob": log_prob, "value": value, "reward": float(reward), "sparse_reward": float(info.get("sparse_reward", reward)), "state_hash": state_hash(obs), "embedding": embedding, "agent_pos": info.get("agent_pos", (0, 0)), "terminated": bool(terminated)})
                    obs = next_obs
                    self.steps += 1
                current, returns = 0.0, []
                for step in reversed(trajectory):
                    current = step["reward"] + 0.99 * current
                    returns.insert(0, current)
                for step, ret in zip(trajectory, returns):
                    step["return"] = ret
                trajectories.append(trajectory)
                env.close()
            return trajectories

        def update(self, trajectories, advantages):
            obs = torch.tensor(np.asarray([s["obs"] for t in trajectories for s in t]), dtype=torch.float32, device=self.device)
            actions = torch.tensor([s["action"] for t in trajectories for s in t], dtype=torch.long, device=self.device)
            old_log_probs = torch.tensor([s["log_prob"] for t in trajectories for s in t], dtype=torch.float32, device=self.device)
            advantage = torch.tensor([a for group in advantages for a in group], dtype=torch.float32, device=self.device)
            returns = torch.tensor([s["return"] for t in trajectories for s in t], dtype=torch.float32, device=self.device)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8) if advantage.numel() > 1 else advantage
            grad_finite, losses, kls = True, [], []
            for _ in range(4):
                for indices in torch.randperm(len(actions), device=self.device).split(min(512, len(actions))):
                    logits, values = self.policy(obs[indices])
                    distribution = Categorical(logits=logits)
                    log_probs = distribution.log_prob(actions[indices])
                    ratio = torch.exp(log_probs - old_log_probs[indices])
                    clipped = torch.clamp(ratio, 0.8, 1.2) * advantage[indices]
                    policy_loss = -torch.minimum(ratio * advantage[indices], clipped).mean()
                    value_loss = F.mse_loss(values.squeeze(-1), returns[indices])
                    loss = policy_loss + 0.5 * value_loss - 0.01 * distribution.entropy().mean()
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    for p in self.policy.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            grad_finite = False
                    self.optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                    kls.append(float((old_log_probs[indices] - log_probs).mean().detach().cpu()))
            self.last_update = {"loss": float(np.mean(losses)) if losses else 0.0, "approx_kl": float(np.mean(kls)) if kls else 0.0, "grad_finite": bool(grad_finite)}
            return self.last_update

        def evaluate(self, episodes=None):
            # Sparse reporting on held-out seeds: dense shaping is training-only.
            episodes = episodes or self.eval_episodes
            successes, returns, lengths = 0, [], []
            self.policy.eval()
            self._eval_count += 1
            for epi in range(episodes):
                eval_seed = self.fixed_seed if self.fixed_seed is not None else self.eval_base_seed + self._eval_count * 100_000 + epi
                env = make_env(self.env_name, eval_seed, False)
                obs, _ = env.reset()
                total, length, terminated = 0.0, 0, False
                while not terminated and length < 1000:
                    with torch.no_grad():
                        logits, _ = self.policy(torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
                    obs, reward, terminated, truncated, _ = env.step(logits.argmax(-1).item())
                    total += reward
                    length += 1
                    if truncated:
                        break
                successes += int(terminated)
                returns.append(total)
                lengths.append(length)
                env.close()
            return {"success_rate": successes / episodes, "mean_return": float(np.mean(returns)), "mean_length": float(np.mean(lengths))}

        def run(self, max_steps=None, progress=None):
            target, next_eval, start = int(max_steps or self.total_steps), self.eval_interval, time.time()
            while self.steps < target:
                trajectories = self.collect()
                advantages = self.method.compute(trajectories)
                update_stats = self.update(trajectories, advantages)
                train_success = sum(1 for t in trajectories if t and t[-1].get("terminated", False)) / max(1, len(trajectories))
                train_ret = sum(sum(s["reward"] for s in t) for t in trajectories) / max(1, len(trajectories))
                if not bool(update_stats.get("grad_finite", True)):
                    print("WARNING: non-finite gradients detected; stopping early (P0 smoke check).")
                    break
                if self.steps >= next_eval:
                    metrics = self.evaluate()
                    metrics.update({"step": self.steps, "elapsed_sec": time.time() - start, "train_success": round(float(train_success), 3), "train_mean_return": round(float(train_ret), 3), **update_stats})
                    if isinstance(self.method, SAHCA):
                        metrics.update(self.method.last_stats)
                    self.entries.append(metrics)
                    if progress:
                        progress(metrics)
                    next_eval += self.eval_interval
            final = self.evaluate()
            final.update({"step": self.steps, "elapsed_sec": time.time() - start, "final": True, **self.last_update})
            if isinstance(self.method, SAHCA):
                final.update(self.method.last_stats)
            self.entries.append(final)
            return self.entries

        def save(self, prefix="sahca"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            # Resolve save dir next to the notebook file when possible;
            # molab kernels often run with cwd != notebook dir, so a bare
            # relative path lands somewhere invisible in the file panel.
            try:
                _base = Path(__file__).parent
            except NameError:
                _base = Path.cwd()
            result_path = (_base / f"{prefix}_{self.method_name.lower()}_seed{self.seed}_{stamp}.json").resolve()
            checkpoint_path = (_base / f"{prefix}_{self.method_name.lower()}_seed{self.seed}_{stamp}.pt").resolve()
            result_path.write_text(json.dumps({"config": self.config, "method": self.method_name, "seed": self.seed, "device": str(self.device), "dense_training": bool(self.dense_reward), "eval_reward": "sparse", "entries": self.entries}, indent=2))
            torch.save({"model": self.policy.state_dict(), "optimizer": self.optimizer.state_dict(), "step": self.steps, "config": self.config, "method": self.method_name, "seed": self.seed}, checkpoint_path)
            print(f"[save] cwd={Path.cwd().resolve()}")
            print(f"[save] results -> {result_path}")
            print(f"[save] checkpoint -> {checkpoint_path}")
            return result_path, checkpoint_path

    return METHODS, Policy, SAHCA, Trainer, make_env


@app.cell
def controls(mo):
    method = mo.ui.dropdown(options=["SAHCA", "GRPO", "GiGPO", "GAGPO"], value="SAHCA", label="Method")
    environment = mo.ui.dropdown(options={"DoorKey 6x6 fixed (P1)": "MiniGrid-DoorKey-6x6-v0|42|True", "DoorKey 6x6 randomized dense-train (P2)": "MiniGrid-DoorKey-6x6-v0|None|True", "DoorKey 6x6 randomized SPARSE (main benchmark)": "MiniGrid-DoorKey-6x6-v0|None|False", "Empty 8x8 smoke (P0)": "MiniGrid-Empty-8x8-v0|None|False", "DoorKey 8x8 (P4, GPU only)": "MiniGrid-DoorKey-8x8-v0|None|True"}, value="DoorKey 6x6 fixed (P1)", label="Environment")
    preset = mo.ui.dropdown(options={"micro_only (validated: a=0,b=1,g=0)": "0.0|1.0|0.0", "macro_only (=GRPO)": "1.0|0.0|0.0", "no_hindsight": "1.0|1.0|0.0", "low_hindsight": "1.0|1.0|0.3", "full_mix": "1.0|1.0|1.0"}, value="micro_only (validated: a=0,b=1,g=0)", label="SAHCA preset (P3 ablation)")
    # GPU-ideal defaults (RTX 6000, molab): 300K steps, 64 eps/update, 50 eval eps.
    # P0 smoke: set steps=10000 + Empty 8x8. P1: 50000 fixed. P2: 300000 randomized. P4: 1000000 DoorKey-8x8.
    steps = mo.ui.number(start=1000, stop=1000000, value=300000, step=10000, label="Environment steps (P0:10K, P1:50K, P2:300K, P4:1M)")
    group_size = mo.ui.number(start=4, stop=128, value=64, step=4, label="Episodes/update (GPU:64, CPU debug:16)")
    eval_episodes = mo.ui.number(start=5, stop=100, value=50, step=5, label="Evaluation episodes (paper:50-100)")
    threshold = mo.ui.number(start=0.5, stop=0.999, value=0.9, step=0.05, label="SAHCA threshold")
    seed = mo.ui.number(start=0, stop=100000, value=0, step=1, label="Seed (run 0-4 for paper)")
    run = mo.ui.run_button(label="Run experiment", kind="success")
    mo.vstack([mo.hstack([method, environment, seed]), mo.hstack([steps, group_size, eval_episodes, threshold]), mo.hstack([preset]), run])
    return environment, eval_episodes, group_size, method, preset, run, seed, steps, threshold


@app.cell
def run_experiment(environment, eval_episodes, group_size, method, preset, run, seed, steps, threshold, Trainer, device, mo):
    entries = []
    result_path = None
    checkpoint_path = None
    history = {}  # label -> entries, accumulated across runs in this session
    if not run.value:
        mo.md("Choose settings and click **Run experiment**. P0 smoke: Empty-8x8 5K steps. Then DoorKey-6x6 fixed 50K GRPO vs SAHCA-micro, same seed. Eval is sparse on held-out seeds.")
    else:
        _env_name, _seed_text, _dense_text = environment.value.split("|")
        _fixed_seed = None if _seed_text == "None" else int(_seed_text)
        _alpha, _beta, _gamma = [float(x) for x in preset.value.split("|")]
        _total_steps = int(steps.value)
        _config = {"env": {"name": _env_name, "fixed_seed": _fixed_seed, "dense_reward": _dense_text == "True"}, "training": {"group_size": int(group_size.value), "total_env_steps": _total_steps, "eval_interval": max(2000, _total_steps // 10), "eval_episodes": int(eval_episodes.value)}, "policy": {"obs_dim": 151, "hidden_dim": 256, "action_dim": 7}, "credit_assignment": {"similarity_threshold": float(threshold.value), "alpha": _alpha, "beta": _beta, "gamma": _gamma, "cross_trajectory_only": True}}
        _trainer = Trainer(_config, method.value, device, int(seed.value))
        print(f"Training: {method.value} | {_env_name} | dense_train={_dense_text} | sparse_eval | steps={_total_steps:,} | seed={int(seed.value)} | device={device} | a={_alpha},b={_beta},g={_gamma}")
        mo.md(f"Running `{method.value}` on `{_env_name}` for `{_total_steps:,}` steps (sparse eval, held-out seeds).")
        def _update_status(_metrics):
            _cov = _metrics.get("soft_anchor_coverage", None)
            _extra = f", coverage={_cov:.1%}" if _cov is not None else ""
            print(f"Step {_metrics['step']:,}: success={_metrics['success_rate']:.1%}, return={_metrics['mean_return']:.3f}, loss={_metrics.get('loss', 0):.3f}, kl={_metrics.get('approx_kl', 0):.4f}{_extra}")
        entries = _trainer.run(max_steps=_total_steps, progress=_update_status)
        result_path, checkpoint_path = _trainer.save()
        _final = entries[-1]
        _label = f"{method.value}-seed{int(seed.value)}"
        history[_label] = entries
        # Learning-curve AUC + steps-to-50% for publication standard.
        import numpy as _np
        _xs = _np.asarray([e["step"] for e in entries], dtype=float)
        _srs = _np.asarray([e["success_rate"] for e in entries], dtype=float)
        _auc = float(_np.trapezoid(_srs, _xs) / max(1.0, _xs[-1] - _xs[0])) if len(_xs) > 1 else float(_srs[-1]) if len(_srs) else 0.0
        try:
            _t50 = next(e["step"] for e in entries if e["success_rate"] >= 0.5)
        except StopIteration:
            _t50 = None
        mo.vstack([mo.md(f"## Run complete `{_label}`\n\n**Device:** `{device}`  \n**Final success (sparse, held-out):** `{_final['success_rate']:.1%}`  \n**Final return:** `{_final['mean_return']:.3f}`  \n**AUC:** `{_auc:.3f}`  \n**Steps to 50%:** `{_t50}`  \n**Results:** `{result_path}`  \n**Checkpoint:** `{checkpoint_path}`"), mo.ui.table(entries)])
    return checkpoint_path, entries, history, result_path


@app.cell
def results_plot(entries, history, mo, np, plt):
    if not entries and not history:
        mo.md("## Learning curves\n\nThe chart appears after the first run. Run GRPO then SAHCA with the same seed to overlay fairly.")
    else:
        _series = dict(history) if history else {}
        if entries and not _series:
            _series["current"] = entries
        _figure, _axes = plt.subplots(1, 2, figsize=(12, 4))
        for _lbl, _ents in _series.items():
            _xs = [e["step"] for e in _ents]
            _axes[0].plot(_xs, [e["success_rate"] for e in _ents], marker="o", label=_lbl)
            _axes[1].plot(_xs, [e["mean_return"] for e in _ents], marker="o", label=_lbl)
        _axes[0].set_title("Success rate (sparse, held-out)")
        _axes[0].set_xlabel("Environment steps")
        _axes[0].set_ylim(-0.05, 1.05)
        _axes[0].grid(alpha=0.3)
        _axes[0].legend(fontsize=8)
        _axes[1].set_title("Mean return (sparse)")
        _axes[1].set_xlabel("Environment steps")
        _axes[1].grid(alpha=0.3)
        _axes[1].legend(fontsize=8)
        _figure.tight_layout()
        mo.mpl.interactive(_figure)


if __name__ == "__main__":
    app.run()
