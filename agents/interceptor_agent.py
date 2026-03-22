import copy
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.snn_network import SNNNetwork
from models.encoder import StateEncoder


class InterceptorAgent:
    """
    MADDPG Interceptor Agent with parameter sharing across interceptors.
    Uses Ornstein-Uhlenbeck noise for exploration.
    Actor: local obs -> SNN -> Linear(3) -> Tanh
    Critic: all agents' obs + actions -> value
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 3,
        n_agents: int = 4,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is None:
            config = {}
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents

        self.lr_actor: float = config.get("lr_actor", 3e-4)
        self.lr_critic: float = config.get("lr_critic", 1e-3)
        self.gamma: float = config.get("gamma", 0.99)
        self.max_grad_norm: float = config.get("max_grad_norm", 0.5)
        self.tau: float = config.get("target_update_tau", 0.01)

        # Actor: encoder -> SNN -> Linear(3) -> Tanh
        self.actor = self._build_actor()
        self.target_actor = copy.deepcopy(self.actor)

        # Centralized critic
        critic_input_dim = (obs_dim + action_dim) * n_agents
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # OU noise state
        self._ou_state = np.zeros(action_dim, dtype=np.float32)
        self._ou_mu = 0.0
        self._ou_theta = 0.15
        self._ou_sigma = 0.2

    def _build_actor(self) -> nn.Module:
        """Build actor network: StateEncoder -> SNN -> Linear(3) -> Tanh."""

        class InterceptorActor(nn.Module):
            def __init__(self, obs_dim: int, action_dim: int) -> None:
                super().__init__()
                self.encoder = StateEncoder(input_dim=obs_dim, output_dim=128)
                self.snn = SNNNetwork(input_dim=128, output_dim=64)
                self.fc_out = nn.Linear(64, action_dim)
                self.tanh = nn.Tanh()

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                enc = self.encoder(obs)
                snn_out = self.snn(enc)
                return self.tanh(self.fc_out(snn_out))

        return InterceptorActor(self.obs_dim, self.action_dim)

    def _ou_noise(self, action: np.ndarray) -> np.ndarray:
        """Apply Ornstein-Uhlenbeck exploration noise."""
        self._ou_state = (
            self._ou_state
            + self._ou_theta * (self._ou_mu - self._ou_state)
            + self._ou_sigma * np.random.randn(*self._ou_state.shape).astype(np.float32)
        )
        return action + self._ou_state

    def get_action(self, obs: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Get continuous thrust vector [x,y,z] in [-1,1]^3.
        Only requires local observation (no global state).
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_tensor).squeeze(0).numpy()

        if add_noise:
            action = self._ou_noise(action)

        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(self, batch: Dict[str, Any]) -> Tuple[float, float]:
        """
        Update actor and critic using MADDPG.
        batch contains: obs (B, n_agents, obs_dim), actions (B, n_agents, action_dim),
                        rewards (B,), next_obs (B, n_agents, obs_dim), dones (B,)
        Returns: actor_loss, critic_loss
        """
        obs = torch.tensor(batch["obs"], dtype=torch.float32)  # (B, n_agents, obs_dim)
        actions = torch.tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.tensor(batch["dones"], dtype=torch.float32)

        B = obs.shape[0]

        # Centralized critic input
        obs_flat = obs.view(B, -1)
        actions_flat = actions.view(B, -1)
        next_obs_flat = next_obs.view(B, -1)

        # Target actions for next obs
        with torch.no_grad():
            next_actions = torch.stack(
                [self.target_actor(next_obs[:, i, :]) for i in range(self.n_agents)],
                dim=1,
            ).view(B, -1)
            target_q = rewards + self.gamma * (1.0 - dones) * self.target_critic(
                torch.cat([next_obs_flat, next_actions], dim=-1)
            ).squeeze(-1)

        current_q = self.critic(torch.cat([obs_flat, actions_flat], dim=-1)).squeeze(-1)
        critic_loss = nn.functional.mse_loss(current_q, target_q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Actor loss
        new_actions = torch.stack(
            [self.actor(obs[:, i, :]) for i in range(self.n_agents)],
            dim=1,
        ).view(B, -1)
        actor_loss = -self.critic(torch.cat([obs_flat, new_actions], dim=-1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        self.sync_target_networks()

        return float(actor_loss.item()), float(critic_loss.item())

    def sync_target_networks(self, tau: Optional[float] = None) -> None:
        """Soft update target networks."""
        if tau is None:
            tau = self.tau
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save(self, path: str) -> None:
        """Save agent state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load agent state."""
        ckpt = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_actor.load_state_dict(ckpt["target_actor"])
        self.target_critic.load_state_dict(ckpt["target_critic"])


if __name__ == "__main__":
    agent = InterceptorAgent(obs_dim=128, action_dim=3, n_agents=4)
    obs = np.random.randn(128).astype(np.float32)
    action = agent.get_action(obs, add_noise=False)
    print(f"Interceptor action: {action}, shape: {action.shape}")
    assert action.shape == (3,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    print("interceptor_agent.py OK")
