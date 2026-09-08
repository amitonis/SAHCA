"""Abstract base class for credit assignment methods."""

from abc import ABC, abstractmethod
from typing import Dict, List


class CreditAssignmentMethod(ABC):
    """Base class for step-level credit assignment in GRPO-style training.

    Subclasses must implement compute_advantages() which takes a group of
    trajectories and returns per-step advantages.
    """

    @abstractmethod
    def compute_advantages(
        self, trajectory_group: List[List[Dict]]
    ) -> List[List[float]]:
        """Compute per-step advantages for a group of trajectories.

        Args:
            trajectory_group: List of N trajectories. Each trajectory is a list
                of step dicts with keys: obs, action, reward, log_prob, value,
                state_hash, embedding, agent_pos, terminated, return.

        Returns:
            List of N lists of floats, matching the structure of trajectory_group.
            Each float is the advantage for the corresponding step.
        """
        ...
