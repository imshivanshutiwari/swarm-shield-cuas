import os
import copy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Individual Q-network for a single drone agent."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MixingNetwork(nn.Module):
    """
    QMIX mixing network.
    Takes per-agent Q-values and global state to produce total Q.
    Monotonicity constraint enforced via absolute weights.
    """

    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim

        # Hypernetworks to generate mixing weights from state
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_qs: (B, n_agents) per-agent Q-values
            state: (B, state_dim) global state
        Returns:
            total_q: (B, 1)
        """
        B = agent_qs.shape[0]
        agent_qs = agent_qs.unsqueeze(1)  # (B, 1, n_agents)

        # First mixing layer with absolute weights (monotonicity)
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(B, 1, self.embed_dim)
        hidden = torch.bmm(agent_qs, w1) + b1  # (B, 1, embed_dim)
        hidden = torch.nn.functional.elu(hidden)

        # Second mixing layer with absolute weights
        w2 = torch.abs(self.hyper_w2(state)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        total_q = torch.bmm(hidden, w2) + b2  # (B, 1, 1)
        return total_q.squeeze(-1)  # (B, 1)


# Available attacker tactics
TACTICS = [
    "maintain_formation",
    "diverge",
    "converge_on_target",
    "activate_jamming",
    "kamikaze_rush",
    "orbit_and_scan",
]


class QMIXAttacker:
    """
    QMIX-based attacker controlling the enemy swarm cooperatively.
    Uses per-drone-type Q-networks and a mixing network.
    """

    def __init__(
        self,
        n_agents: int = 10,
        obs_dim: int = 64,
        action_dim: int = 6,
        config: Optional[Dict[str, Any]] = None,
        state_dim: int = 128,
    ) -> None:
        if config is None:
            config = {}
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma: float = config.get("gamma", 0.99)
        self.lr: float = config.get("lr_critic", 1e-3)
        self.max_grad_norm: float = config.get("max_grad_norm", 0.5)

        # Per-agent Q-networks (parameter sharing across all drones)
        self.q_network = QNetwork(obs_dim, action_dim)
        self.target_q_network = copy.deepcopy(self.q_network)

        # Mixing network
        self.mixing_network = MixingNetwork(n_agents, state_dim)
        self.target_mixing_network = copy.deepcopy(self.mixing_network)

        all_params = list(self.q_network.parameters()) + list(self.mixing_network.parameters())
        self.optimizer = optim.Adam(all_params, lr=self.lr)

        self._tactic_history: List[str] = []
        self._tactic_probs: Dict[str, float] = {t: 1.0 / len(TACTICS) for t in TACTICS}

    def get_actions(self, obs_list: List[np.ndarray]) -> List[int]:
        """
        Get discrete actions for each drone agent.
        Actions index into TACTICS list.
        """
        actions = []
        for obs in obs_list:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.q_network(obs_t).squeeze(0)
            # Epsilon-greedy with epsilon=0.1
            if np.random.rand() < 0.1:
                action = int(np.random.randint(0, self.action_dim))
            else:
                action = int(q_vals.argmax().item())
            actions.append(action)
            if 0 <= action < len(TACTICS):
                self._tactic_history.append(TACTICS[action])
        return actions

    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute QMIX TD loss."""
        obs = torch.tensor(batch["obs"], dtype=torch.float32)
        actions = torch.tensor(batch["actions"], dtype=torch.long)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.tensor(batch["dones"], dtype=torch.float32)
        state = torch.tensor(batch["state"], dtype=torch.float32)
        next_state = torch.tensor(batch["next_state"], dtype=torch.float32)

        B = obs.shape[0]

        # Current Q values (per agent)
        q_vals = self.q_network(obs.view(B * self.n_agents, -1)).view(B, self.n_agents, -1)
        chosen_q = q_vals.gather(2, actions.view(B, self.n_agents, 1)).squeeze(-1)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_q_network(next_obs.view(B * self.n_agents, -1)).view(
                B, self.n_agents, -1
            )
            next_chosen_q = next_q.max(dim=-1).values

            target_total_q = self.target_mixing_network(next_chosen_q, next_state)
            targets = (
                rewards.unsqueeze(-1) + self.gamma * (1.0 - dones.unsqueeze(-1)) * target_total_q
            )

        # Mix current Q values
        total_q = self.mixing_network(chosen_q, state)
        td_loss = nn.functional.mse_loss(total_q, targets.detach())
        return td_loss

    def update_mixing_network(self, batch: Dict[str, Any]) -> float:
        """Update Q-networks and mixing network."""
        loss = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q_network.parameters()) + list(self.mixing_network.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()

        # Soft update target networks
        tau = 0.01
        for tp, p in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
        for tp, p in zip(self.target_mixing_network.parameters(), self.mixing_network.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

        return float(loss.item())

    def mutate_tactics(self, diversity_score: float) -> None:
        """
        If diversity is low, force a new combination of tactics.
        Diversity score is entropy of tactic distribution in [0, log(n_tactics)].
        """
        min_diversity = np.log(len(TACTICS)) * 0.3  # 30% of max entropy
        if diversity_score < min_diversity:
            # Reset to uniform distribution and force exploration
            self._tactic_probs = {t: 1.0 / len(TACTICS) for t in TACTICS}
            # Increase exploration temporarily
            self._tactic_history = []

    def save(self, path: str) -> None:
        """Save agent state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "mixing_network": self.mixing_network.state_dict(),
                "target_q_network": self.target_q_network.state_dict(),
                "target_mixing_network": self.target_mixing_network.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load agent state."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.q_network.load_state_dict(ckpt["q_network"])
        self.mixing_network.load_state_dict(ckpt["mixing_network"])
        
        if "target_q_network" in ckpt:
            self.target_q_network.load_state_dict(ckpt["target_q_network"])
        else:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            
        if "target_mixing_network" in ckpt:
            self.target_mixing_network.load_state_dict(ckpt["target_mixing_network"])
        else:
            self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

    def get_tactic_distribution(self) -> Dict[str, float]:
        """Return probability distribution over tactics from recent history."""
        if not self._tactic_history:
            return dict(self._tactic_probs)
        counts: Dict[str, int] = {t: 0 for t in TACTICS}
        for t in self._tactic_history[-100:]:
            if t in counts:
                counts[t] += 1
        total = sum(counts.values())
        if total == 0:
            return dict(self._tactic_probs)
        return {t: c / total for t, c in counts.items()}


if __name__ == "__main__":
    attacker = QMIXAttacker(n_agents=10, obs_dim=64, action_dim=6)
    obs_list = [np.random.randn(64).astype(np.float32) for _ in range(10)]
    actions = attacker.get_actions(obs_list)
    print(f"Attacker actions: {actions}")
    print(f"Tactic distribution: {attacker.get_tactic_distribution()}")
    print("attacker_agent.py OK")
