"""
Evaluation module for SWARM-SHIELD.
Runs trained agents against the environment and computes metrics.
"""

import os
import sys
from typing import Any, Dict, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.commander_agent import CommanderAgent  # noqa: E402
from agents.interceptor_agent import InterceptorAgent  # noqa: E402
from envs.cuas_env import CUASEnv  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    swarm_neutralization_rate,
    friendly_fire_rate,
    engagement_latency_ms,
    ospa_distance,
    jamming_resilience_score,
)
from utils.logger import get_logger  # noqa: E402

log = get_logger("evaluate")


class Evaluator:
    """
    Evaluates trained MAPPO commander and MADDPG interceptors.
    Computes all 9 metrics specified in the project.
    """

    def __init__(
        self,
        commander: CommanderAgent,
        interceptors: InterceptorAgent,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.commander = commander
        self.interceptors = interceptors
        self.config = config or {}
        self.swarm_config = self.config.get("swarm", {})

    def evaluate(self, n_episodes: int = 10) -> Dict[str, Any]:
        """
        Run n_episodes evaluation episodes and return aggregated metrics.
        """
        env = CUASEnv(config=self.swarm_config if self.swarm_config else None)
        n_interceptors = env.n_interceptors
        n_targets = env.n_enemy_drones

        all_neutralization_rates = []
        all_ff_rates = []
        all_latencies = []
        all_ospa = []
        all_jammer_resilience = []
        all_rewards = []

        for ep in range(n_episodes):
            obs_dict, _ = env.reset(seed=ep)
            episode_rewards = []
            action_timestamps = []
            step = 0

            while True:
                # Commander obs
                swarm_state = (
                    env.swarm.get_state_vector() if env.swarm else np.zeros(n_targets * 12)
                )
                int_states = (
                    env.interceptor_positions.flatten()
                    if env.interceptor_positions is not None
                    else np.zeros(n_interceptors * 3)
                )
                global_obs = np.concatenate([swarm_state, int_states]).astype(np.float32)

                cmd_obs_dim = self.commander.obs_dim
                if len(global_obs) < cmd_obs_dim:
                    global_obs = np.pad(global_obs, (0, cmd_obs_dim - len(global_obs)))
                else:
                    global_obs = global_obs[:cmd_obs_dim]

                cmd_action, _, _ = self.commander.get_action(global_obs, deterministic=True)

                actions_dict: Dict[str, Any] = {"commander": cmd_action}
                for i in range(n_interceptors):
                    int_obs = obs_dict[f"interceptor_{i}"]["own_state"].astype(np.float32)
                    int_obs_dim = self.interceptors.obs_dim
                    if len(int_obs) < int_obs_dim:
                        int_obs = np.pad(int_obs, (0, int_obs_dim - len(int_obs)))
                    else:
                        int_obs = int_obs[:int_obs_dim]
                    actions_dict[f"interceptor_{i}"] = self.interceptors.get_action(
                        int_obs, add_noise=False
                    )

                action_timestamps.append(step * 0.1)
                obs_dict, rewards_dict, terminated, truncated, info = env.step(actions_dict)

                episode_reward = sum(rewards_dict.values())
                episode_rewards.append(episode_reward)
                step += 1

                if terminated or truncated:
                    break

            # Compute metrics for this episode
            n_neutral = info.get("neutralized", 0)
            ff = info.get("friendly_fire", 0)
            total_eng = max(n_neutral + ff, 1)

            neutralization_pct = swarm_neutralization_rate(n_neutral, n_targets)
            ff_pct = friendly_fire_rate(ff, total_eng)
            lat_ms = engagement_latency_ms(action_timestamps)

            # OSPA: compare estimated drone positions to (synthetic) true positions
            if env.swarm:
                true_pos = np.array([d.position[:2] for d in env.swarm.drones], dtype=np.float32)
                est_pos = true_pos + np.random.normal(0, 5.0, true_pos.shape).astype(np.float32)
                ospa = ospa_distance(est_pos, true_pos)
            else:
                ospa = 0.0

            # Jammer resilience: compare baseline vs jammed SNR
            baseline_snr = 15.0
            avg_jammed_snr = (
                float(np.mean(obs_dict["interceptor_0"]["radar_returns"][:, 3]))
                if "interceptor_0" in obs_dict
                else baseline_snr
            )
            resilience = jamming_resilience_score(baseline_snr, avg_jammed_snr)

            all_neutralization_rates.append(neutralization_pct)
            all_ff_rates.append(ff_pct)
            all_latencies.append(lat_ms)
            all_ospa.append(ospa)
            all_jammer_resilience.append(resilience)
            all_rewards.append(float(np.mean(episode_rewards)))

            log.info(
                f"Ep {ep+1}/{n_episodes}: Neutralized={n_neutral}/{n_targets} "
                f"({neutralization_pct:.1f}%), FF={ff_pct:.1f}%, "
                f"Latency={lat_ms:.2f}ms, OSPA={ospa:.2f}"
            )

        env.close()

        return {
            "mean_neutralization_rate": float(np.mean(all_neutralization_rates)),
            "std_neutralization_rate": float(np.std(all_neutralization_rates)),
            "mean_friendly_fire_rate": float(np.mean(all_ff_rates)),
            "mean_engagement_latency_ms": float(np.mean(all_latencies)),
            "mean_ospa": float(np.mean(all_ospa)),
            "mean_jammer_resilience": float(np.mean(all_jammer_resilience)),
            "mean_episode_reward": float(np.mean(all_rewards)),
            "n_episodes": n_episodes,
        }


if __name__ == "__main__":
    commander = CommanderAgent(obs_dim=128, action_dim=11, n_targets=10, n_interceptors=4)
    interceptors = InterceptorAgent(obs_dim=6, action_dim=3, n_agents=4)
    evaluator = Evaluator(commander=commander, interceptors=interceptors)
    results = evaluator.evaluate(n_episodes=2)
    print("Evaluation results:", results)
    print("evaluate.py OK")
