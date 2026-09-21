"""PPO Trainer with rollout collection, evaluation, and JSON logging."""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.env import make_env, state_hash
from src.policy import CNNPolicy
from methods import get_method


class Trainer:
    """PPO trainer for MiniGrid credit assignment experiments.

    Args:
        config: Dict with training/env/policy/credit_assignment settings.
        method_name: Name of credit assignment method ('grpo', 'gigpo', 'gagpo', 'sahca').
        device: Torch device ('cuda' or 'cpu').
        seed: Random seed for reproducibility.
        log_dir: Directory for JSON result files.
    """

    # PPO hyperparameters
    CLIP_EPS = 0.2
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    PPO_EPOCHS = 4
    LR = 3e-4
    GAMMA = 0.99

    def __init__(
        self,
        config: Dict,
        method_name: str,
        device: str = "auto",
        seed: int = 0,
        log_dir: str = "logs",
    ):
        self.config = config
        self.method_name = method_name
        self.seed = seed
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Seed everything
        torch.manual_seed(seed)
        np.random.seed(seed)
        self._train_rng = np.random.default_rng(seed)
        # Held-out eval base is offset so train and eval layouts never overlap
        # (PLAN rule: keep training and evaluation seeds separate).
        self.eval_base_seed = 1_000_000 + seed
        self._eval_count = 0

        # Environment
        env_config = config.get("env", {})
        self.env_name = env_config.get("name", "MiniGrid-DoorKey-6x6-v0")
        self.fixed_seed = env_config.get("fixed_seed", None)
        self.use_dense_reward = env_config.get("dense_reward", True)

        # Training params
        train_config = config.get("training", {})
        self.group_size = train_config.get("group_size", 64)
        self.total_env_steps = train_config.get("total_env_steps", 300000)
        self.eval_episodes = train_config.get("eval_episodes", 100)
        self.eval_interval = train_config.get("eval_interval", 10000)

        # Policy
        policy_config = config.get("policy", {})
        self.hidden_dim = policy_config.get("hidden_dim", 256)
        obs_dim = policy_config.get("obs_dim", 151)
        action_dim = policy_config.get("action_dim", 7)
        self.policy = CNNPolicy(
            obs_dim=obs_dim, hidden_dim=self.hidden_dim, action_dim=action_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.LR)

        # Credit assignment method
        ca_config = config.get("credit_assignment", {})
        self.method = get_method(method_name, **ca_config)

        # Logging
        self.log_entries: List[Dict] = []
        self.total_steps_collected = 0

    def collect_rollouts(self) -> List[List[Dict]]:
        """Collect a group of complete episode trajectories.

        Returns:
            List of `group_size` trajectories. Each trajectory is a list of step dicts
            with keys: obs, action, reward, log_prob, value, state_hash, embedding,
            agent_pos, terminated.
        """
        trajectories = []
        self.policy.eval()

        for epi in range(self.group_size):
            # Per-episode seed: fixed layout reuses the same map (intended for
            # the "fixed" benchmark); randomized layouts draw a fresh seed per
            # episode from the train RNG so group diversity is reproducible.
            if self.fixed_seed is not None:
                episode_seed = self.fixed_seed
            else:
                episode_seed = int(self._train_rng.integers(0, 2**31 - 1))
            env = make_env(
                self.env_name,
                fixed_seed=episode_seed,
                use_dense_reward=self.use_dense_reward,
            )
            obs, info = env.reset()
            trajectory = []
            done = False
            sparse_total = 0.0

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action, log_prob, value, embedding = self.policy.get_action(obs_tensor)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                sparse_total += float(info.get("sparse_reward", reward))

                step = {
                    "obs": obs.copy() if isinstance(obs, np.ndarray) else np.array(obs),
                    "action": action,
                    "reward": float(reward),
                    "sparse_reward": float(info.get("sparse_reward", reward)),
                    "log_prob": float(log_prob),
                    "value": float(value),
                    "state_hash": state_hash(np.asarray(obs)),
                    "embedding": embedding.copy(),
                    "agent_pos": info.get("agent_pos", (0, 0)),
                    "terminated": bool(terminated),
                }
                trajectory.append(step)
                obs = next_obs
                self.total_steps_collected += 1

            # Compute discounted returns for each step
            returns = self._compute_returns(trajectory)
            for i, ret in enumerate(returns):
                trajectory[i]["return"] = ret

            trajectories.append(trajectory)
            env.close()

        return trajectories

    def _compute_returns(self, trajectory: List[Dict]) -> List[float]:
        """Compute discounted returns from each step to end of episode.

        Args:
            trajectory: List of step dicts with 'reward' key.

        Returns:
            List of discounted returns, one per step.
        """
        returns = []
        G = 0.0
        for step in reversed(trajectory):
            G = step["reward"] + self.GAMMA * G
            returns.insert(0, G)
        return returns

    def ppo_update(self, trajectories: List[List[Dict]], advantages: List[List[float]]):
        """Perform PPO policy update using collected trajectories and computed advantages.

        Args:
            trajectories: List of trajectories (list of step dicts).
            advantages: Matching per-step advantages, same shape as trajectories.

        Returns:
            Dict with mean loss / policy loss / value loss / entropy / KL stats,
            plus a finite-gradient flag (PLAN P0 smoke check).
        """
        self.policy.train()

        # Flatten all steps
        all_obs = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []

        for traj, traj_advs in zip(trajectories, advantages):
            for step, adv in zip(traj, traj_advs):
                all_obs.append(step["obs"])
                all_actions.append(step["action"])
                all_old_log_probs.append(step["log_prob"])
                all_advantages.append(adv)
                all_returns.append(step["return"])

        # Convert to tensors
        obs_t = torch.tensor(np.array(all_obs), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(all_actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(all_old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(all_advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(all_returns, dtype=torch.float32, device=self.device)

        # Normalize advantages
        if advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # PPO epochs
        n_samples = len(all_obs)
        batch_size = min(512, n_samples)
        loss_hist, pl_hist, vl_hist, ent_hist, kl_hist = [], [], [], [], []
        grad_finite = True

        for _ in range(self.PPO_EPOCHS):
            # Shuffle indices
            indices = torch.randperm(n_samples, device=self.device)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]

                batch_obs = obs_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_old_log_probs = old_log_probs_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                batch_returns = returns_t[batch_idx]

                # Evaluate current policy
                log_probs, entropy, values = self.policy.evaluate_actions(
                    batch_obs, batch_actions
                )

                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.CLIP_EPS, 1.0 + self.CLIP_EPS) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.VALUE_COEF * value_loss
                    + self.ENTROPY_COEF * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.MAX_GRAD_NORM)
                # P0 smoke check: gradients must stay finite.
                for p in self.policy.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grad_finite = False
                self.optimizer.step()

                loss_hist.append(float(loss.detach().cpu()))
                pl_hist.append(float(policy_loss.detach().cpu()))
                vl_hist.append(float(value_loss.detach().cpu()))
                ent_hist.append(float(entropy.mean().detach().cpu()))
                kl_hist.append(float((batch_old_log_probs - log_probs).mean().detach().cpu()))

        import numpy as _np
        return {
            "loss": float(_np.mean(loss_hist)) if loss_hist else 0.0,
            "policy_loss": float(_np.mean(pl_hist)) if pl_hist else 0.0,
            "value_loss": float(_np.mean(vl_hist)) if vl_hist else 0.0,
            "entropy": float(_np.mean(ent_hist)) if ent_hist else 0.0,
            "approx_kl": float(_np.mean(kl_hist)) if kl_hist else 0.0,
            "grad_finite": bool(grad_finite),
        }

    def evaluate(self, num_episodes: Optional[int] = None) -> Dict:
        """Evaluate the current policy without exploration.

        Evaluation always uses the sparse MiniGrid reward (dense shaping is
        training-only) and held-out seeds so train/eval never overlap.

        Args:
            num_episodes: Number of evaluation episodes (defaults to config).

        Returns:
            Dict with 'success_rate', 'mean_return' (sparse), 'mean_length'.
        """
        if num_episodes is None:
            num_episodes = self.eval_episodes

        self.policy.eval()
        successes = 0
        total_returns = []
        total_lengths = []
        self._eval_count += 1

        for epi in range(num_episodes):
            # Held-out eval seed: never equal to any train seed.
            if self.fixed_seed is not None:
                # Fixed benchmark: same map is intended, but offset the counter
                # so the eval stream is explicit and reproducible.
                episode_seed = self.fixed_seed
            else:
                episode_seed = self.eval_base_seed + self._eval_count * 100_000 + epi
            env = make_env(
                self.env_name,
                fixed_seed=episode_seed,
                use_dense_reward=False,  # sparse reporting
            )
            obs, info = env.reset()
            episode_return = 0.0
            episode_length = 0
            done = False

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    action_logits, _ = self.policy(obs_tensor.unsqueeze(0))
                action = action_logits.argmax(dim=-1).item()

                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                episode_length += 1
                done = terminated or truncated

            if terminated:
                successes += 1
            total_returns.append(episode_return)
            total_lengths.append(episode_length)
            env.close()

        return {
            "success_rate": successes / num_episodes,
            "mean_return": float(np.mean(total_returns)),
            "mean_length": float(np.mean(total_lengths)),
        }

    def train(self, max_steps: Optional[int] = None) -> List[Dict]:
        """Main training loop.

        Args:
            max_steps: Override total_env_steps from config.

        Returns:
            List of log entries (also saved to JSON).
        """
        total_steps = max_steps if max_steps is not None else self.total_env_steps
        self.total_steps_collected = 0
        iteration = 0
        start_time = time.time()

        log_filename = (
            f"{self.env_name}_{self.method_name}_seed{self.seed}.json"
        ).replace("/", "_")
        log_path = self.log_dir / log_filename

        pbar = tqdm(total=total_steps, desc=f"{self.method_name} seed={self.seed}")
        last_update_stats: Dict = {}

        while self.total_steps_collected < total_steps:
            # Collect rollouts
            trajectories = self.collect_rollouts()
            steps_in_group = sum(len(t) for t in trajectories)

            # Compute credit-assigned advantages
            advantages = self.method.compute_advantages(trajectories)

            # PPO update
            update_stats = self.ppo_update(trajectories, advantages)
            last_update_stats = update_stats
            # Training-stream stats (stochastic policy): visible even when
            # greedy eval is still 0 early in training.
            train_success = float(sum(1 for t in trajectories if t and t[-1].get("terminated", False)) / max(1, len(trajectories)))
            train_dense_ret = float(sum(sum(s["reward"] for s in t) for t in trajectories) / max(1, len(trajectories)))

            iteration += 1
            pbar.update(steps_in_group)

            # Evaluate periodically
            if self.total_steps_collected >= (len(self.log_entries) + 1) * self.eval_interval:
                eval_results = self.evaluate()
                elapsed = time.time() - start_time
                steps_per_sec = self.total_steps_collected / elapsed if elapsed > 0 else 0

                entry = {
                    "step": self.total_steps_collected,
                    "iteration": iteration,
                    "elapsed_sec": round(elapsed, 1),
                    "steps_per_sec": round(steps_per_sec, 1),
                    "train_success": round(train_success, 3),
                    "train_mean_return": round(train_dense_ret, 3),
                    **eval_results,
                    **update_stats,
                }
                if hasattr(self.method, "last_stats"):
                    try:
                        entry.update({f"sahca_{k}": v for k, v in self.method.last_stats.items()})
                    except Exception:
                        pass
                self.log_entries.append(entry)

                tqdm.write(
                    f"[Step {self.total_steps_collected:>7d}] "
                    f"SR={eval_results['success_rate']:.2f} "
                    f"Ret={eval_results['mean_return']:.3f} "
                    f"Len={eval_results['mean_length']:.1f} "
                    f"loss={update_stats['loss']:.3f} kl={update_stats['approx_kl']:.4f} "
                    f"grad_finite={update_stats['grad_finite']} "
                    f"({steps_per_sec:.0f} steps/s)"
                )

                # Save incrementally
                self._save_log(log_path)

        pbar.close()

        # Final evaluation
        final_eval = self.evaluate()
        elapsed = time.time() - start_time
        final_entry = {
            "step": self.total_steps_collected,
            "iteration": iteration,
            "elapsed_sec": round(elapsed, 1),
            "steps_per_sec": round(self.total_steps_collected / elapsed, 1) if elapsed > 0 else 0,
            "final": True,
            **final_eval,
            **last_update_stats,
        }
        if hasattr(self.method, "last_stats"):
            try:
                final_entry.update({f"sahca_{k}": v for k, v in self.method.last_stats.items()})
            except Exception:
                pass
        self.log_entries.append(final_entry)
        self._save_log(log_path)

        print(f"\n{'='*60}")
        print(f"Training complete: {self.method_name} seed={self.seed}")
        print(f"  Total steps: {self.total_steps_collected}")
        print(f"  Final SR:    {final_eval['success_rate']:.2f}")
        print(f"  Final Ret:   {final_eval['mean_return']:.3f}")
        print(f"  Time:        {elapsed:.1f}s")
        print(f"  Log:         {log_path}")
        print(f"{'='*60}\n")

        return self.log_entries

    def _save_log(self, path: Path):
        """Save log entries to JSON file."""
        result = {
            "method": self.method_name,
            "seed": self.seed,
            "config": self.config,
            "device": str(self.device),
            "code_version": self._git_hash(),
            "dense_training": bool(self.use_dense_reward),
            "eval_reward": "sparse",
            "entries": self.log_entries,
        }
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

    @staticmethod
    def _git_hash() -> str:
        try:
            import subprocess
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], text=True, timeout=5
            ).strip()
        except Exception:
            return "unknown"

