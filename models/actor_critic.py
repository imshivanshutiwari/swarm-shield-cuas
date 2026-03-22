from typing import Tuple

import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    """Actor network for policy output."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticNetwork(nn.Module):
    """Critic (value) network."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """
    Combined actor-critic module for PPO/MAPPO-style algorithms.
    The actor outputs action logits and the critic outputs a state value.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        is_continuous: bool = True,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.is_continuous = is_continuous

        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim)
        self.critic = CriticNetwork(obs_dim, hidden_dim)

        if is_continuous:
            # Learnable log std for continuous actions
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_logits: (B, action_dim) — mean for continuous, logits for discrete
            value: (B, 1)
            log_std: (action_dim,) — only for continuous
        """
        logits = self.actor(obs)
        value = self.critic(obs)
        if self.is_continuous:
            return logits, value, self.log_std
        return logits, value, torch.zeros_like(logits)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return state value estimate."""
        return self.critic(obs)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log_prob, entropy, and value for given obs/actions.
        Returns:
            log_probs: (B,)
            entropy: scalar
            values: (B, 1)
        """
        logits, values, log_std = self.forward(obs)

        if self.is_continuous:
            std = log_std.exp()
            dist = torch.distributions.Normal(logits, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
        else:
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

        return log_probs, entropy, values


class DDPGActorCritic(nn.Module):
    """
    Actor-Critic for MADDPG with separate actor and centralized critic.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents

        # Decentralized actor
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # continuous actions in [-1, 1]
        )

        # Centralized critic: takes all agents' obs + actions
        critic_input_dim = (obs_dim + action_dim) * n_agents
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic action from actor."""
        return self.actor(obs)

    def get_value(self, all_obs: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        """
        Centralized value estimate.
        Args:
            all_obs: (B, n_agents * obs_dim)
            all_actions: (B, n_agents * action_dim)
        """
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.critic(x)


if __name__ == "__main__":
    ac = ActorCritic(obs_dim=128, action_dim=4, is_continuous=False)
    x = torch.randn(8, 128)
    logits, value, _ = ac(x)
    print(f"Logits: {logits.shape}, Value: {value.shape}")
    print("actor_critic.py OK")
