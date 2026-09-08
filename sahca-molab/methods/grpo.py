"""GRPO: Group Relative Policy Optimization.

Every step in a trajectory receives the same advantage = the trajectory's
z-scored return (relative to the group).
"""

from typing import Dict, List

import numpy as np

from methods.base import CreditAssignmentMethod


class GRPO(CreditAssignmentMethod):
    """Standard GRPO baseline -- trajectory-level credit assignment only.

    Z-scores trajectory total returns across the group, then assigns
    every step in a trajectory the same advantage.
    """

    def compute_advantages(
        self, trajectory_group: List[List[Dict]]
    ) -> List[List[float]]:
        # Compute total return per trajectory
        traj_returns = []
        for traj in trajectory_group:
            total_return = sum(step["reward"] for step in traj)
            traj_returns.append(total_return)

        # Z-score across the group
        traj_returns = np.array(traj_returns, dtype=np.float64)
        mean = traj_returns.mean()
        std = traj_returns.std()
        if std < 1e-8:
            z_scored = np.zeros_like(traj_returns)
        else:
            z_scored = (traj_returns - mean) / std

        # Assign same advantage to every step in each trajectory
        advantages = []
        for i, traj in enumerate(trajectory_group):
            adv = float(z_scored[i])
            advantages.append([adv] * len(traj))

        return advantages
