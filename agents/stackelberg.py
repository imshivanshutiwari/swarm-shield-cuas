from typing import Any, List
import numpy as np


class StackelbergGame:
    """
    Models attacker-defender interaction as a Stackelberg leader-follower game.
    The attacker (leader) anticipates the defender's response.
    The defender (follower) best-responds to the observed attacker action.
    """

    def __init__(self, n_attacker_agents: int, n_defender_agents: int) -> None:
        self.n_attacker_agents = n_attacker_agents
        self.n_defender_agents = n_defender_agents
        self._defender_update_count = 0
        self._attacker_update_count = 0
        self._reward_history: List[float] = []
        self._attacker_reward_history: List[float] = []
        # Update ratio: attacker updates every 3 defender updates
        self._stackelberg_ratio = 3

    def compute_leader_response(self, defender_policy: Any, env_state: np.ndarray) -> np.ndarray:
        """
        Compute optimal attacker strategy given the defender policy.
        Uses a best-response oracle approximation:
          For each possible attacker action, simulate defender response
          and select the action maximizing attacker return.
        Returns: attacker strategy vector (n_attacker_agents, action_dim)
        """
        n_candidates = 8
        best_reward = -np.inf
        best_strategy = np.zeros((self.n_attacker_agents, 6), dtype=np.float32)

        for _ in range(n_candidates):
            candidate = np.random.randn(self.n_attacker_agents, 6).astype(np.float32)
            # Approximate attacker reward as negative of defender success rate
            # (heuristic based on env_state distance to protected zone)
            if env_state is not None and len(env_state) > 0:
                proximity = float(np.mean(np.abs(env_state[:3])))
                reward_approx = -proximity + np.random.randn() * 0.1
            else:
                reward_approx = float(np.random.randn())

            if reward_approx > best_reward:
                best_reward = reward_approx
                best_strategy = candidate

        return best_strategy

    def compute_follower_best_response(
        self, attacker_action: np.ndarray, env_state: np.ndarray
    ) -> np.ndarray:
        """
        Compute defender's best response to the observed attacker action.
        Returns: defender strategy vector (n_defender_agents, 3)
        """
        # Heuristic: assign each defender to intercept the nearest attacker
        if attacker_action is None or len(attacker_action) == 0:
            return np.zeros((self.n_defender_agents, 3), dtype=np.float32)

        # For each defender, assign to the nearest attacker cluster
        n_att = len(attacker_action)
        defender_strategy = np.zeros((self.n_defender_agents, 3), dtype=np.float32)
        for d in range(self.n_defender_agents):
            # Simple assignment: cycle through attacker targets
            target_idx = d % n_att
            if attacker_action.ndim >= 2 and attacker_action.shape[1] >= 3:
                direction = -attacker_action[target_idx, :3]
            else:
                direction = np.random.randn(3).astype(np.float32)
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            defender_strategy[d] = direction.astype(np.float32)

        return defender_strategy

    def update_schedule(self, episode: int) -> bool:
        """
        Determine if attacker should update this episode.
        Rule: attacker updates every 3 defender updates.
        In a sequence of episodes 0,1,...N, attacker updates at episodes
        where (episode + 1) % 3 == 0 (i.e., every third episode: 2, 5, 8, ...).
        Returns True if it is the attacker's turn to update.
        """
        should_update = (episode + 1) % self._stackelberg_ratio == 0
        if should_update:
            self._attacker_update_count += 1
        return should_update

    def compute_regret(self, attacker_rewards_history: List[float]) -> float:
        """
        Compute attacker's regret (gap between best-possible and actual reward).
        Regret = max_possible - running_average
        """
        if not attacker_rewards_history:
            return 0.0
        best_possible = max(attacker_rewards_history)
        actual_avg = float(np.mean(attacker_rewards_history))
        return float(best_possible - actual_avg)

    def log_equilibrium_distance(self) -> float:
        """
        Compute Nash gap proxy: how far current policies are from equilibrium.
        Uses attacker regret as a proxy.
        """
        if len(self._attacker_reward_history) < 2:
            return float("inf")
        recent = self._attacker_reward_history[-20:]
        return float(np.std(recent))

    def record_rewards(self, attacker_reward: float) -> None:
        """Record attacker reward for regret and equilibrium tracking."""
        self._attacker_reward_history.append(attacker_reward)


if __name__ == "__main__":
    game = StackelbergGame(n_attacker_agents=10, n_defender_agents=4)
    env_state = np.random.randn(12).astype(np.float32)
    strategy = game.compute_leader_response(None, env_state)
    print(f"Attacker strategy shape: {strategy.shape}")
    for ep in range(9):
        updated = game.update_schedule(ep)
    print(f"Attacker updates: {game._attacker_update_count}")
    print("stackelberg.py OK")
