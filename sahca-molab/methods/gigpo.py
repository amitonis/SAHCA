"""GiGPO: Groups steps with exact same state hash across trajectories.

Within each hash group, z-scores downstream returns to produce micro-advantages.
Steps with no hash match fall back to GRPO-style trajectory advantage.

Reference: works well on fixed layouts, collapses on randomized (-75% SR drop).
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np

from methods.base import CreditAssignmentMethod


class GiGPO(CreditAssignmentMethod):
    """GiGPO: exact-state grouping with z-scored micro-advantages.

    Groups steps by state_hash across trajectories, then z-scores
    downstream returns within each hash group.
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

        # Compute GRPO fallback: trajectory-level z-scored returns
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

        # Initialize advantages with GRPO fallback
        advantages = []
        for i, traj in enumerate(trajectory_group):
            advantages.append([float(z_traj[i])] * len(traj))

        # Override with micro-advantages for steps in hash groups with 2+ members
        for state_h, members in hash_groups.items():
            if len(members) < 2:
                continue

            # Collect downstream returns for this hash group
            returns = np.array([m[2] for m in members], dtype=np.float64)
            mean_r = returns.mean()
            std_r = returns.std()

            if std_r < 1e-8:
                z_returns = np.zeros_like(returns)
            else:
                z_returns = (returns - mean_r) / std_r

            # Assign micro-advantages
            for (traj_idx, step_idx, _), z_val in zip(members, z_returns):
                advantages[traj_idx][step_idx] = float(z_val)

        return advantages
