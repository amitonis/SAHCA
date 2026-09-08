"""CNN policy network for MiniGrid environments.

Architecture:
    Grid input (7x7x3 = 147) + direction one-hot (4) = 151 total obs_dim.
    Conv2d(3,32,3,pad=1) -> ReLU -> Conv2d(32,64,3,pad=1) -> ReLU -> Flatten
    -> Linear(cnn_out+4, hidden_dim) -> ReLU -> Linear(hidden_dim, hidden_dim) -> ReLU
    Actor: Linear(hidden_dim, action_dim)
    Critic: Linear(hidden_dim, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class CNNPolicy(nn.Module):
    """CNN policy with actor-critic heads and embedding extraction.

    Args:
        obs_dim: Total flat observation dimension (default 151).
        hidden_dim: Hidden layer size (default 256 for GPU).
        action_dim: Number of discrete actions (default 7 for MiniGrid).
    """

    def __init__(self, obs_dim: int = 151, hidden_dim: int = 256, action_dim: int = 7):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Grid dimensions: 7x7 partial view, 3 channels (object, color, state)
        self.grid_size = 7
        self.grid_channels = 3
        self.grid_flat = self.grid_size * self.grid_size * self.grid_channels  # 147
        self.dir_dim = obs_dim - self.grid_flat  # 4

        # CNN for grid observation
        self.conv1 = nn.Conv2d(self.grid_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # After conv with padding=1 on 7x7: output is still 7x7
        cnn_out_dim = 64 * self.grid_size * self.grid_size  # 64*7*7 = 3136

        # FC layers combining CNN output + direction
        self.fc1 = nn.Linear(cnn_out_dim + self.dir_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor and critic heads
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for stable RL training."""
        for module in [self.conv1, self.conv2, self.fc1, self.fc2]:
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def _extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract hidden representation from flat observation.

        Args:
            obs: Flat observation tensor of shape (batch, obs_dim).

        Returns:
            Hidden features of shape (batch, hidden_dim).
        """
        batch_size = obs.shape[0]

        # Split into grid and direction
        grid_flat = obs[:, :self.grid_flat]
        direction = obs[:, self.grid_flat:]

        # Reshape grid: (batch, 147) -> (batch, 3, 7, 7)
        grid = grid_flat.reshape(batch_size, self.grid_size, self.grid_size, self.grid_channels)
        grid = grid.permute(0, 3, 1, 2)  # (batch, channels, H, W)

        # CNN forward
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        x = x.reshape(batch_size, -1)  # Flatten CNN output

        # Concatenate with direction and pass through FC layers
        x = torch.cat([x, direction], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

    def forward(self, obs: torch.Tensor):
        """Forward pass returning action logits and value estimate.

        Args:
            obs: Flat observation tensor of shape (batch, obs_dim).

        Returns:
            Tuple of (action_logits, value) with shapes (batch, action_dim) and (batch, 1).
        """
        features = self._extract_features(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract 256-dim embedding for SAHCA clustering.

        Args:
            obs: Flat observation tensor of shape (batch, obs_dim).

        Returns:
            Detached embedding tensor of shape (batch, hidden_dim).
        """
        with torch.no_grad():
            features = self._extract_features(obs)
        return features.detach()

    def get_action(self, obs: torch.Tensor):
        """Sample an action and return all needed rollout data.

        Args:
            obs: Single observation tensor of shape (obs_dim,) or (1, obs_dim).

        Returns:
            Tuple of (action, log_prob, value, embedding).
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        features = self._extract_features(obs)
        action_logits = self.actor(features)
        value = self.critic(features)

        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        embedding = features.detach().squeeze(0).cpu().numpy()

        return (
            action.item(),
            log_prob.item(),
            value.item(),
            embedding,
        )

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate log probs and entropy for given obs-action pairs (for PPO update).

        Args:
            obs: Batch of observations, shape (batch, obs_dim).
            actions: Batch of actions, shape (batch,).

        Returns:
            Tuple of (log_probs, entropy, values) all shape (batch,) or (batch, 1).
        """
        features = self._extract_features(obs)
        action_logits = self.actor(features)
        values = self.critic(features)

        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values.squeeze(-1)
