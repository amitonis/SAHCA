"""MiniGrid environment wrapper with dense reward shaping.

Observation pipeline:
    MiniGrid dict obs -> ImgObsWrapper (7x7x3 = 147) -> flatten + direction one-hot (4) = 151
"""

import hashlib
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from minigrid.wrappers import ImgObsWrapper


def state_hash(obs: np.ndarray) -> str:
    """MD5 hash of observation bytes, truncated to 16 chars."""
    return hashlib.md5(obs.tobytes()).hexdigest()[:16]


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Manhattan distance between two grid positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class FlatDirObsWrapper(gym.ObservationWrapper):
    """Flattens the 7x7x3 image and appends a 4-dim direction one-hot.

    MiniGrid's ImgObsWrapper gives (7,7,3) uint8 image.
    This wrapper produces a flat float32 vector of size 147 + 4 = 151.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # 7*7*3 image + 4 direction one-hot
        self.obs_dim = 7 * 7 * 3 + 4  # 151
        self.observation_space = gym.spaces.Box(
            low=0.0, high=255.0, shape=(self.obs_dim,), dtype=np.float32
        )

    def observation(self, obs):
        # obs is (7,7,3) from ImgObsWrapper
        flat_img = obs.flatten().astype(np.float32)

        # Direction one-hot
        direction = self.unwrapped.agent_dir  # 0-3
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[direction] = 1.0

        return np.concatenate([flat_img, dir_onehot])


class DenseRewardWrapper(gym.Wrapper):
    """Wraps MiniGrid envs to add dense reward shaping for DoorKey tasks.

    Reward components:
        - step_penalty: -0.01 per step
        - key_pickup: +0.5 (once per episode)
        - door_open: +0.5 (once per episode)
        - distance_shaping: 0.1 * (prev_manhattan - curr_manhattan) to goal
        - terminal: MiniGrid default (+1.0 - 0.9*(steps/max_steps))
    """

    def __init__(self, env: gym.Env, use_dense_reward: bool = True):
        super().__init__(env)
        self.use_dense_reward = use_dense_reward
        self._has_key = False
        self._door_opened = False
        self._prev_dist_to_goal = None
        self._goal_pos = None
        self._step_count = 0

    def _find_goal_pos(self):
        """Find the goal position in the full grid."""
        grid = self.unwrapped.grid
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "goal":
                    return (x, y)
        return None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._has_key = False
        self._door_opened = False
        self._step_count = 0

        self._goal_pos = self._find_goal_pos()
        agent_pos = self.unwrapped.agent_pos
        if self._goal_pos is not None:
            self._prev_dist_to_goal = manhattan_distance(agent_pos, self._goal_pos)
        else:
            self._prev_dist_to_goal = None

        info["agent_pos"] = tuple(agent_pos)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._step_count += 1

        agent_pos = tuple(self.unwrapped.agent_pos)
        info["agent_pos"] = agent_pos

        if not self.use_dense_reward:
            return obs, reward, terminated, truncated, info

        dense_reward = 0.0

        # Step penalty
        dense_reward += -0.01

        # Key pickup bonus (once)
        if not self._has_key and self.unwrapped.carrying is not None:
            if hasattr(self.unwrapped.carrying, "type") and self.unwrapped.carrying.type == "key":
                self._has_key = True
                dense_reward += 0.5

        # Door open bonus (once) -- check if any door cell is now open
        if not self._door_opened:
            grid = self.unwrapped.grid
            for x in range(grid.width):
                for y in range(grid.height):
                    cell = grid.get(x, y)
                    if cell is not None and cell.type == "door" and cell.is_open:
                        self._door_opened = True
                        dense_reward += 0.5
                        break
                if self._door_opened:
                    break

        # Distance shaping toward goal
        if self._goal_pos is not None and self._prev_dist_to_goal is not None:
            curr_dist = manhattan_distance(agent_pos, self._goal_pos)
            dense_reward += 0.1 * (self._prev_dist_to_goal - curr_dist)
            self._prev_dist_to_goal = curr_dist

        # Terminal reward: keep MiniGrid's default reward on success
        if terminated:
            dense_reward += reward  # MiniGrid gives 1 - 0.9*(steps/max_steps)

        return obs, dense_reward, terminated, truncated, info


def make_env(
    env_name: str = "MiniGrid-DoorKey-6x6-v0",
    fixed_seed: Optional[int] = None,
    use_dense_reward: bool = True,
) -> gym.Env:
    """Create a MiniGrid environment with flat obs (151-dim) and optional dense reward.

    Wrapping order: gym.make -> ImgObsWrapper -> FlatDirObsWrapper -> DenseRewardWrapper -> FixedSeedWrapper

    Args:
        env_name: MiniGrid environment ID.
        fixed_seed: If set, reset always uses this seed (fixed layout).
                    If None, layouts are randomized each reset.
        use_dense_reward: Whether to apply DenseRewardWrapper.

    Returns:
        Wrapped gymnasium environment. Observation is flat (151,).
    """
    env = gym.make(env_name)
    env = ImgObsWrapper(env)       # Dict obs -> (7,7,3) image
    env = FlatDirObsWrapper(env)   # (7,7,3) -> flat (151,) with direction one-hot

    if use_dense_reward:
        env = DenseRewardWrapper(env, use_dense_reward=True)

    if fixed_seed is not None:
        env = FixedSeedWrapper(env, seed=fixed_seed)

    return env


class FixedSeedWrapper(gym.Wrapper):
    """Forces env.reset() to always use the same seed for fixed layouts."""

    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env)
        self._fixed_seed = seed

    def reset(self, **kwargs):
        kwargs["seed"] = self._fixed_seed
        return super().reset(**kwargs)
