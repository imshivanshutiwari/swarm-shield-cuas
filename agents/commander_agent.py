import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.snn_network import SNNNetwork
from models.encoder import StateEncoder


class CommanderAgent:
    """
    MAPPO Commander Agent with centralized critic.
    The commander assigns interceptors to targets using a multi-discrete action space.
    Actor uses SNN-encoder + Linear heads.
    Critic uses a global MLP taking full swarm state.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[Dict[str, Any]] = None,
        n_targets: int = 10,
        n_interceptors: int = 4,
    ) -> None:
        if config is None:
            config = {}
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_targets = n_targets
        self.n_interceptors = n_interceptors

        self.lr_actor: float = config.get("lr_actor", 3e-4)
        self.lr_critic: float = config.get("lr_critic", 1e-3)
        self.gamma: float = config.get("gamma", 0.99)
        self.gae_lambda: float = config.get("gae_lambda", 0.95)
        self.clip_epsilon: float = config.get("clip_epsilon", 0.2)
        self.entropy_coef: float = config.get("entropy_coef", 0.01)
        self.max_grad_norm: float = config.get("max_grad_norm", 0.5)
        self.n_epochs: int = config.get("n_epochs", 10)

        self.actor = self._build_actor_network()
        self.critic = self._build_critic_network()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def _build_actor_network(self) -> nn.Module:
        """SNN encoder + Linear heads for commander actor."""

        class CommanderActor(nn.Module):
            def __init__(self, obs_dim: int, n_interceptors: int, n_targets: int) -> None:
                super().__init__()
                self.encoder = StateEncoder(input_dim=obs_dim, output_dim=128)
                self.snn = SNNNetwork(input_dim=128, output_dim=64)
                # One head per interceptor (assigns to target)
                self.heads = nn.ModuleList(
                    [nn.Linear(64, n_targets + 1) for _ in range(n_interceptors)]
                )

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                enc = self.encoder(obs)
                snn_out = self.snn(enc)
                logits = torch.stack([head(snn_out) for head in self.heads], dim=1)
                return logits  # (B, n_interceptors, n_targets+1)

        return CommanderActor(self.obs_dim, self.n_interceptors, self.n_targets)

    def _build_critic_network(self) -> nn.Module:
        """MLP taking global state (full swarm + all interceptors)."""
        return nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def get_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Sample commander action.
        Returns:
            action: (n_interceptors,) int assignment array
            log_prob: float
            value: float
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(obs_tensor)  # (1, n_interceptors, n_targets+1)
            value = self.critic(obs_tensor).squeeze().item()

        assignments = []
        log_prob = 0.0

        for i in range(self.n_interceptors):
            logit = logits[0, i]
            dist = torch.distributions.Categorical(logits=logit)
            if deterministic:
                a = logit.argmax().item()
            else:
                a = dist.sample().item()
            log_prob += dist.log_prob(torch.tensor(a)).item()
            assignments.append(a)

        return np.array(assignments, dtype=int), log_prob, value

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = 0.0
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update(self, rollout_buffer: Any) -> Tuple[float, float, float]:
        """
        Update commander using MAPPO (clipped PPO with centralized critic).
        Returns: policy_loss, value_loss, entropy
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.n_epochs):
            obs_batch = torch.tensor(rollout_buffer.obs, dtype=torch.float32)
            actions_batch = torch.tensor(rollout_buffer.actions, dtype=torch.long)
            old_log_probs_batch = torch.tensor(rollout_buffer.log_probs, dtype=torch.float32)
            advantages_batch = torch.tensor(rollout_buffer.advantages, dtype=torch.float32)
            returns_batch = torch.tensor(rollout_buffer.returns, dtype=torch.float32)

            # Normalize advantages
            advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                advantages_batch.std() + 1e-8
            )

            # Actor forward
            logits = self.actor(obs_batch)  # (B, n_interceptors, n_targets+1)
            new_log_probs = torch.zeros(obs_batch.shape[0])
            entropy_total = torch.tensor(0.0)

            for i in range(self.n_interceptors):
                dist = torch.distributions.Categorical(logits=logits[:, i, :])
                lp = dist.log_prob(
                    actions_batch[:, i] if actions_batch.dim() > 1 else actions_batch
                )
                new_log_probs = new_log_probs + lp
                entropy_total = entropy_total + dist.entropy().mean()

            entropy_mean = entropy_total / self.n_interceptors
            ratio = torch.exp(new_log_probs - old_log_probs_batch)
            surr1 = ratio * advantages_batch
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                * advantages_batch
            )
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy_mean

            # Critic forward
            values = self.critic(obs_batch).squeeze(-1)
            value_loss = nn.functional.mse_loss(values, returns_batch)

            # Backprop
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_mean.item()

        n = self.n_epochs
        return total_policy_loss / n, total_value_loss / n, total_entropy / n

    def save(self, path: str) -> None:
        """Save actor and critic state dicts."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load actor and critic state dicts."""
        ckpt = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])


if __name__ == "__main__":
    commander = CommanderAgent(obs_dim=128, action_dim=11, n_targets=10, n_interceptors=4)
    obs = np.random.randn(128).astype(np.float32)
    action, lp, val = commander.get_action(obs)
    print(f"Commander action: {action}, log_prob: {lp:.3f}, value: {val:.3f}")
    print("commander_agent.py OK")
