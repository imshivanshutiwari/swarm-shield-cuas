import os
from typing import Any, Dict, List, Tuple

import numpy as np

from adversarial.attacker_agent import QMIXAttacker
from envs.cuas_env import CUASEnv


def _safe_wandb_log(metrics: Dict[str, Any], step: int) -> None:
    """Log to WandB if available, otherwise silently skip."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass


class DigitalTwin:
    """
    Adversarial self-play training loop.
    Coordinates the commander, interceptors, and QMIX attacker
    in a closed-loop adversarial training regime.
    """

    def __init__(
        self,
        env_config: Dict[str, Any],
        attacker: QMIXAttacker,
        commander: Any,
        interceptors: Any,
        diversity_threshold: float = 0.5,
    ) -> None:
        self.env_config = env_config
        self.attacker = attacker
        self.commander = commander
        self.interceptors = interceptors
        self.diversity_threshold = diversity_threshold
        self.env = CUASEnv(config=env_config)
        self._episode_count = 0
        self._tactic_history: List[str] = []

    def run_episode(self) -> Tuple[float, float, Dict[str, Any]]:
        """
        Run one full episode of adversarial self-play.
        Returns:
            attacker_reward: total attacker reward
            defender_reward: total defender (commander+interceptors) reward
            metrics: dict with episode statistics
        """
        obs, info = self.env.reset()
        total_attacker_reward = 0.0
        total_defender_reward = 0.0
        step = 0

        # Get swarm obs for attacker
        swarm = self.env.swarm

        while True:
            # Attacker selects tactics for each drone
            attacker_obs_list = [
                (
                    np.concatenate([d.position, d.velocity, [d.health, float(d.is_alive)]]).astype(
                        np.float32
                    )[: self.attacker.obs_dim]
                    if len(np.concatenate([d.position, d.velocity, [d.health, float(d.is_alive)]]))
                    >= self.attacker.obs_dim
                    else np.pad(
                        np.concatenate([d.position, d.velocity, [d.health, float(d.is_alive)]]),
                        (
                            0,
                            self.attacker.obs_dim
                            - len(
                                np.concatenate(
                                    [d.position, d.velocity, [d.health, float(d.is_alive)]]
                                )
                            ),
                        ),
                    ).astype(np.float32)
                )
                for d in (swarm.alive_drones if swarm else [])
            ]

            # Fill up if fewer than n_agents alive
            while len(attacker_obs_list) < self.attacker.n_agents:
                attacker_obs_list.append(np.zeros(self.attacker.obs_dim, dtype=np.float32))

            self.attacker.get_actions(
                attacker_obs_list[: self.attacker.n_agents]
            )
            tactic_dist = self.attacker.get_tactic_distribution()

            # Commander assigns interceptors
            swarm_state = (
                self.env.swarm.get_state_vector()
                if self.env.swarm
                else np.zeros(self.env.n_enemy_drones * 12)
            )
            int_states = (
                self.env.interceptor_positions.flatten()
                if self.env.interceptor_positions is not None
                else np.zeros(self.env.n_interceptors * 3)
            )
            global_obs = np.concatenate([swarm_state, int_states]).astype(np.float32)
            # Pad / truncate to commander obs_dim
            cmd_obs_dim = self.commander.obs_dim
            if len(global_obs) < cmd_obs_dim:
                global_obs = np.pad(global_obs, (0, cmd_obs_dim - len(global_obs)))
            else:
                global_obs = global_obs[:cmd_obs_dim]

            commander_action, _, _ = self.commander.get_action(global_obs)

            # Interceptors select thrust
            actions_dict: Dict[str, Any] = {"commander": commander_action}
            for i in range(self.env.n_interceptors):
                int_obs = obs[f"interceptor_{i}"]["own_state"]
                # Pad to interceptor obs_dim
                int_obs_dim = self.interceptors.obs_dim
                if len(int_obs) < int_obs_dim:
                    int_obs = np.pad(int_obs, (0, int_obs_dim - len(int_obs)))
                else:
                    int_obs = int_obs[:int_obs_dim]
                actions_dict[f"interceptor_{i}"] = self.interceptors.get_action(
                    int_obs.astype(np.float32), add_noise=True
                )

            obs, rewards, terminated, truncated, info = self.env.step(actions_dict)
            total_defender_reward += sum(
                v for k, v in rewards.items() if k.startswith("interceptor_")
            )
            total_attacker_reward += (
                -total_defender_reward * 0.1
            )  # Adversarial: attacker gains when defender loses
            step += 1

            if terminated or truncated:
                break

            # Refresh alive drones reference
            swarm = self.env.swarm

        metrics = {
            "attacker_reward": total_attacker_reward,
            "defender_reward": total_defender_reward,
            "n_neutralized": info.get("neutralized", 0),
            "steps": step,
            "tactic_distribution": tactic_dist,
        }
        self._episode_count += 1
        return total_attacker_reward, total_defender_reward, metrics

    def compute_tactic_diversity(self, history: List[str]) -> float:
        """
        Compute entropy of tactic distribution over recent history.
        Returns float in [0, log(n_tactics)].
        """
        if not history:
            return 0.0
        from adversarial.attacker_agent import TACTICS

        counts = {t: 0 for t in TACTICS}
        for t in history:
            if t in counts:
                counts[t] += 1
        total = sum(counts.values())
        if total == 0:
            return 0.0
        probs = np.array([c / total for c in counts.values() if c > 0])
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        return entropy

    def run_adversarial_training(self, n_episodes: int) -> None:
        """
        Full adversarial self-play training loop.
        Follows Stackelberg schedule: attacker updates every 3 defender updates.
        """
        from agents.stackelberg import StackelbergGame

        stackelberg = StackelbergGame(
            n_attacker_agents=self.attacker.n_agents,
            n_defender_agents=self.env.n_interceptors,
        )

        for episode in range(n_episodes):
            attacker_reward, defender_reward, metrics = self.run_episode()

            # Update defender
            # (In full training, would use rollout buffer — simplified here)

            # Update attacker on Stackelberg schedule
            should_update_attacker = stackelberg.update_schedule(episode)
            if should_update_attacker:
                # Build a minimal dummy batch for attacker update
                B = 4
                dummy_batch = {
                    "obs": np.random.randn(B, self.attacker.n_agents, self.attacker.obs_dim).astype(
                        np.float32
                    ),
                    "actions": np.random.randint(
                        0, self.attacker.action_dim, (B, self.attacker.n_agents)
                    ),
                    "rewards": np.random.randn(B).astype(np.float32),
                    "next_obs": np.random.randn(
                        B, self.attacker.n_agents, self.attacker.obs_dim
                    ).astype(np.float32),
                    "dones": np.zeros(B, dtype=np.float32),
                    "state": np.random.randn(B, self.attacker.state_dim).astype(np.float32),
                    "next_state": np.random.randn(B, self.attacker.state_dim).astype(np.float32),
                }
                self.attacker.update_mixing_network(dummy_batch)

            # Compute tactic diversity
            diversity = self.compute_tactic_diversity(self.attacker._tactic_history[-100:])
            if diversity < self.diversity_threshold:
                self.attacker.mutate_tactics(diversity)

            # Log to WandB
            nash_gap = stackelberg.log_equilibrium_distance()
            log_metrics = {
                "attacker_reward": attacker_reward,
                "defender_reward": defender_reward,
                "tactic_diversity": diversity,
                "nash_gap": nash_gap,
                "episode": episode,
            }
            log_metrics.update(metrics.get("tactic_distribution", {}))
            _safe_wandb_log(log_metrics, step=episode)

    def save_checkpoint(
        self, episode: int, checkpoint_dir: str = "checkpoints/digital_twin"
    ) -> None:
        """Save training checkpoint."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        import torch

        torch.save(
            {
                "episode": episode,
                "attacker_q_network": self.attacker.q_network.state_dict(),
                "attacker_mixing_network": self.attacker.mixing_network.state_dict(),
            },
            os.path.join(checkpoint_dir, f"dt_ep{episode:06d}.pt"),
        )

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        import torch

        ckpt = torch.load(path, map_location="cpu")
        self.attacker.q_network.load_state_dict(ckpt["attacker_q_network"])
        self.attacker.mixing_network.load_state_dict(ckpt["attacker_mixing_network"])


if __name__ == "__main__":
    from agents.commander_agent import CommanderAgent
    from agents.interceptor_agent import InterceptorAgent

    config = {
        "grid_size": 500,
        "n_enemy_drones": 3,
        "drone_types": {
            "kamikaze": {"count": 2, "speed": 40, "health": 1.0},
            "isr": {"count": 1, "speed": 15, "health": 1.5},
            "jammer": {"count": 0, "speed": 8, "health": 2.0},
        },
        "n_interceptors": 4,
        "max_timesteps": 10,
        "n_jammers": 0,
        "jammer_radius": 80,
        "jammer_power_dbm": 45,
        "gps_denial_zones": 0,
        "communication_radius": 150,
        "radar_range": 300,
        "eo_ir_range": 200,
    }
    attacker = QMIXAttacker(n_agents=3, obs_dim=64, action_dim=6, state_dim=128)
    commander = CommanderAgent(obs_dim=128, action_dim=11, n_targets=3, n_interceptors=4)
    interceptors = InterceptorAgent(obs_dim=128, action_dim=3, n_agents=4)
    dt = DigitalTwin(config, attacker, commander, interceptors)
    att_r, def_r, metrics = dt.run_episode()
    print(f"Attacker reward: {att_r:.2f}, Defender reward: {def_r:.2f}")
    print("digital_twin.py OK")
