from typing import Dict

import numpy as np


class RolloutBuffer:
    """
    Stores rollout data for on-policy learning (MAPPO).
    Supports batched sampling and GAE computation.
    """

    def __init__(
        self,
        rollout_steps: int = 2048,
        obs_dim: int = 128,
        action_dim: int = 4,
        n_agents: int = 4,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents

        self.obs = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(rollout_steps, dtype=np.float32)
        self.rewards = np.zeros(rollout_steps, dtype=np.float32)
        self.values = np.zeros(rollout_steps, dtype=np.float32)
        self.dones = np.zeros(rollout_steps, dtype=np.float32)
        self.advantages = np.zeros(rollout_steps, dtype=np.float32)
        self.returns = np.zeros(rollout_steps, dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: float,
    ) -> None:
        """Add a single transition to the buffer."""
        idx = self._ptr % self.rollout_steps
        self.obs[idx] = (
            obs[: self.obs_dim]
            if len(obs) >= self.obs_dim
            else np.pad(obs, (0, self.obs_dim - len(obs)))
        )
        self.actions[idx] = (
            action[: self.action_dim]
            if len(action) >= self.action_dim
            else np.pad(action, (0, self.action_dim - len(action)))
        )
        self.log_probs[idx] = log_prob
        self.rewards[idx] = reward
        self.values[idx] = value
        self.dones[idx] = done
        self._ptr += 1
        self._size = min(self._size + 1, self.rollout_steps)

    def compute_returns_and_advantages(
        self,
        last_value: float = 0.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE advantages and discounted returns in-place."""
        size = self._size
        gae = 0.0
        for t in reversed(range(size)):
            if t == size - 1:
                next_val = last_value
                next_done = 0.0
            else:
                next_val = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = self.rewards[t] + gamma * next_val * (1.0 - next_done) - self.values[t]
            gae = delta + gamma * gae_lambda * (1.0 - next_done) * gae
            self.advantages[t] = gae
            self.returns[t] = self.advantages[t] + self.values[t]

    def get_batch(self, batch_size: int = 256) -> Dict[str, np.ndarray]:
        """Return a random mini-batch from the buffer."""
        size = self._size
        indices = np.random.choice(size, min(batch_size, size), replace=False)
        return {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "log_probs": self.log_probs[indices],
            "rewards": self.rewards[indices],
            "values": self.values[indices],
            "dones": self.dones[indices],
            "advantages": self.advantages[indices],
            "returns": self.returns[indices],
        }

    def reset(self) -> None:
        """Reset buffer pointer and size."""
        self._ptr = 0
        self._size = 0

    @property
    def is_full(self) -> bool:
        """Return True if buffer has enough data for an update."""
        return self._size >= self.rollout_steps

    def __len__(self) -> int:
        return self._size


if __name__ == "__main__":
    buf = RolloutBuffer(rollout_steps=100, obs_dim=64, action_dim=4)
    for i in range(100):
        buf.add(
            obs=np.random.randn(64).astype(np.float32),
            action=np.random.randint(0, 4, 4).astype(np.float32),
            log_prob=float(np.random.randn()),
            reward=float(np.random.randn()),
            value=float(np.random.randn()),
            done=float(i == 99),
        )
    buf.compute_returns_and_advantages()
    batch = buf.get_batch(32)
    print(f"Batch obs shape: {batch['obs'].shape}")
    print("rollout_buffer.py OK")
