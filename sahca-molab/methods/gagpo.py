"""GAGPO: Grouped Advantage with grouped value proxy via TD-error.

Same exact-state grouping as GiGPO, but uses TD-error with a grouped
value proxy (mean return of all steps with matching hash) instead of
z-scored returns.

Reference: achieves 100% on DoorKey-6x6 fixed, but -98% drop on randomized.
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np

from methods.base import CreditAssignmentMethod


class GAGPO(CreditAssignmentMethod):
    """GAGPO: exact-state grouping with TD-error and grouped value proxy.

    Value proxy for a state = mean downstream return of all steps sharing
    that state hash. Advantage = return - value_proxy (TD-error style).
    Falls back to GRPO for steps with no hash match.
    """

    def compute_advantages(
        self, trajectory_group: List[List[Dict]]
    ) -> List[List[float]]:
        # Build hash -> list of (traj_idx, step_idx, return) mappings
        hash_groups: Dict[str, List] = defaultdict(list)
        for traj_idx, traj in enumerate(trajectory_group):
            for step_idx, step in enumerate(traj):
                hash_groups[step["state_hash"]].append(
                    (traj_idx, step_idx, step["return"])
                )

        # Compute grouped value proxy: mean return per hash
        hash_value_proxy: Dict[str, float] = {}
        for state_h, members in hash_groups.items():
            returns = [m[2] for m in members]
            hash_value_proxy[state_h] = float(np.mean(returns))

        # Compute GRPO fallback
        traj_returns = np.array(
            [sum(s["reward"] for s in traj) for traj in trajectory_group],
            dtype=np.float64,
        )
        mean_ret = traj_returns.mean()
        std_ret = traj_returns.std()
        if std_ret < 1e-8:
            z_traj = np.zeros_like(traj_returns)
        else:
            z_traj = (traj_returns - mean_ret) / std_ret

        # Initialize with GRPO fallback
        advantages = []
        for i, traj in enumerate(trajectory_group):
            advantages.append([float(z_traj[i])] * len(traj))

        # Override with TD-error for steps in groups with 2+ members
        for state_h, members in hash_groups.items():
            if len(members) < 2:
                continue

            value_proxy = hash_value_proxy[state_h]

            # TD-error style advantage: return - value_proxy
            td_errors = []
            for traj_idx, step_idx, ret in members:
                td_errors.append(ret - value_proxy)

            # Z-score the TD-errors within the group for stability
            td_errors = np.array(td_errors, dtype=np.float64)
            td_mean = td_errors.mean()
            td_std = td_errors.std()
            if td_std < 1e-8:
                z_td = np.zeros_like(td_errors)
            else:
                z_td = (td_errors - td_mean) / td_std

            for (traj_idx, step_idx, _), z_val in zip(members, z_td):
                advantages[traj_idx][step_idx] = float(z_val)

        return advantages
