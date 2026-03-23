"""
Main trainer orchestrating multi-agent training.
Handles rollout collection, agent updates, and metric logging.
"""

import os
import re
import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agents.commander_agent import CommanderAgent
from agents.interceptor_agent import InterceptorAgent
from adversarial.attacker_agent import QMIXAttacker
from adversarial.curriculum import CurriculumScheduler
from envs.cuas_env import CUASEnv
from training.rollout_buffer import RolloutBuffer
from training.callbacks import CheckpointCallback, WandBCallback
from utils.logger import get_logger

log = get_logger("trainer")


class MARLTrainer:
    """
    Multi-agent reinforcement learning trainer.
    Coordinates MAPPO commander, MADDPG interceptors, and QMIX attacker.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.marl_config = config.get("marl", {})
        self.swarm_config = config.get("swarm", {})
        self.snn_config = config.get("snn", {})
        self.total_timesteps: int = self.marl_config.get("total_timesteps", 3_000_000)
        self.rollout_steps: int = self.marl_config.get("rollout_steps", 2048)
        self.batch_size: int = self.marl_config.get("batch_size", 256)
        self.checkpoint_dir: str = config.get("checkpoint_dir", "checkpoints")

        self.curriculum = CurriculumScheduler()
        self.env: Optional[CUASEnv] = None

        # Agent dimensions
        self.n_interceptors = self.swarm_config.get("n_interceptors", 4)
        self.n_enemy_drones = self.swarm_config.get("n_enemy_drones", 10)
        self.obs_dim = 128
        self.cmd_obs_dim = self.n_enemy_drones * 12 + self.n_interceptors * 6

        # Initialize agents
        self.commander = CommanderAgent(
            obs_dim=self.cmd_obs_dim,
            action_dim=self.n_enemy_drones + 1,
            config=self.marl_config,
            n_targets=self.n_enemy_drones,
            n_interceptors=self.n_interceptors,
        )
        self.interceptors = InterceptorAgent(
            obs_dim=6,  # own_state dim
            action_dim=3,
            n_agents=self.n_interceptors,
            config=self.marl_config,
        )
        self.attacker = QMIXAttacker(
            n_agents=self.n_enemy_drones,
            obs_dim=64,
            action_dim=6,
            config=self.marl_config,
            state_dim=128,
        )

        # Rollout buffers
        self.commander_buffer = RolloutBuffer(
            rollout_steps=self.rollout_steps,
            obs_dim=self.cmd_obs_dim,
            action_dim=self.n_interceptors,
        )
        self.interceptor_buffer = RolloutBuffer(
            rollout_steps=self.rollout_steps,
            obs_dim=6,
            action_dim=3,
        )

        # Callbacks
        self.checkpoint_cb = CheckpointCallback(
            checkpoint_dir=self.checkpoint_dir,
            save_every_n_episodes=100,
            agents={
                "commander": self.commander,
                "interceptors": self.interceptors,
                "attacker": self.attacker,
            },
        )
        self.wandb_cb = WandBCallback(enabled=config.get("use_wandb", False))

        self.global_step: int = 0
        self.episode: int = 0
        self._episode_rewards: List[float] = []

        if self.config.get("resume"):
            self._resume_from_checkpoint()

    def _resume_from_checkpoint(self) -> None:
        """Find and load the latest checkpoint from the checkpoint directory."""
        log.info(f"Resuming from checkpoint in {self.checkpoint_dir}...")
        
        # Look for ckpt_epXXXXXX.pt in any agent subfolder
        # Since they are saved together, finding the latest in 'commander' is enough
        search_path = os.path.join(self.checkpoint_dir, "commander", "ckpt_ep*.pt")
        ckpt_files = glob.glob(search_path)
        
        if not ckpt_files:
            log.warning("No checkpoints found to resume from. Starting from scratch.")
            return

        # Extract episode numbers and find the latest
        episodes = []
        for f in ckpt_files:
            match = re.search(r"ckpt_ep(\d+)\.pt", f)
            if match:
                episodes.append(int(match.group(1)))
        
        if not episodes:
            log.warning("Could not parse episode numbers. Starting from scratch.")
            return

        latest_ep = max(episodes)
        log.info(f"Latest checkpoint found: episode {latest_ep}")

        # Load for all agents
        try:
            self.commander.load(os.path.join(self.checkpoint_dir, "commander", f"ckpt_ep{latest_ep:06d}.pt"))
            self.interceptors.load(os.path.join(self.checkpoint_dir, "interceptors", f"ckpt_ep{latest_ep:06d}.pt"))
            self.attacker.load(os.path.join(self.checkpoint_dir, "attacker", f"ckpt_ep{latest_ep:06d}.pt"))
            
            self.episode = latest_ep
            self.global_step = latest_ep * self.rollout_steps
            log.info(f"Successfully resumed from episode {latest_ep} (step {self.global_step})")
        except Exception as e:
            log.error(f"Failed to load checkpoints: {e}. Starting from scratch.")

    def _init_env(self, env_config: Optional[Dict] = None) -> None:
        """Initialize or re-initialize the environment."""
        if env_config is None:
            env_config = self.curriculum.get_config(self.global_step)
        if self.env is not None:
            self.env.close()
        self.env = CUASEnv(config=env_config)

    def _collect_rollout(self) -> Tuple[float, float, int]:
        """
        Collect rollout_steps of experience.
        Returns:
            mean_commander_reward: float
            mean_interceptor_reward: float
            steps_collected: int
        """
        assert self.env is not None, "Environment not initialized"
        obs_dict, _ = self.env.reset()

        episode_cmd_rewards: List[float] = []
        episode_int_rewards: List[float] = []
        steps = 0

        self.commander_buffer.reset()
        self.interceptor_buffer.reset()

        for _ in range(self.rollout_steps):
            # Commander action
            swarm_state = (
                self.env.swarm.get_state_vector()
                if self.env.swarm
                else np.zeros(self.n_enemy_drones * 12)
            )
            int_states = (
                self.env.interceptor_positions.flatten()
                if self.env.interceptor_positions is not None
                else np.zeros(self.n_interceptors * 3)
            )
            global_obs = np.concatenate([swarm_state, int_states]).astype(np.float32)

            # Pad/truncate
            if len(global_obs) < self.cmd_obs_dim:
                global_obs = np.pad(global_obs, (0, self.cmd_obs_dim - len(global_obs)))
            else:
                global_obs = global_obs[: self.cmd_obs_dim]

            cmd_action, cmd_log_prob, cmd_value = self.commander.get_action(global_obs)

            # Interceptor actions
            actions_dict: Dict[str, Any] = {"commander": cmd_action}
            int_obs_list = []
            for i in range(self.n_interceptors):
                int_obs = obs_dict[f"interceptor_{i}"]["own_state"].astype(np.float32)
                int_obs_list.append(int_obs)
                actions_dict[f"interceptor_{i}"] = self.interceptors.get_action(
                    int_obs, add_noise=True
                )

            obs_next, rewards_dict, terminated, truncated, info = self.env.step(actions_dict)

            # Store transitions
            cmd_reward = rewards_dict.get("commander", 0.0)
            int_reward = (
                sum(rewards_dict.get(f"interceptor_{i}", 0.0) for i in range(self.n_interceptors))
                / self.n_interceptors
            )

            self.commander_buffer.add(
                obs=global_obs,
                action=cmd_action.astype(np.float32),
                log_prob=cmd_log_prob,
                reward=cmd_reward,
                value=cmd_value,
                done=float(terminated or truncated),
            )
            self.interceptor_buffer.add(
                obs=int_obs_list[0],
                action=actions_dict["interceptor_0"],
                log_prob=0.0,
                reward=int_reward,
                value=0.0,
                done=float(terminated or truncated),
            )

            episode_cmd_rewards.append(cmd_reward)
            episode_int_rewards.append(int_reward)
            obs_dict = obs_next
            steps += 1
            self.global_step += 1

            if terminated or truncated:
                break

        self.commander_buffer.compute_returns_and_advantages(
            gamma=self.marl_config.get("gamma", 0.99),
            gae_lambda=self.marl_config.get("gae_lambda", 0.95),
        )

        return (
            float(np.mean(episode_cmd_rewards)) if episode_cmd_rewards else 0.0,
            float(np.mean(episode_int_rewards)) if episode_int_rewards else 0.0,
            steps,
        )

    def train_step(self) -> Dict[str, Any]:
        """Perform one full training cycle."""
        # Get current curriculum config
        curr_config = self.curriculum.get_config(self.global_step)
        if self.env is None or curr_config.get("n_enemy_drones") != self.n_enemy_drones:
            self._init_env(curr_config)

        # Collect rollout
        cmd_reward, int_reward, steps = self._collect_rollout()

        # Update commander (MAPPO)
        if self.commander_buffer.is_full or self.commander_buffer._size > 0:
            policy_loss, value_loss, entropy = self.commander.update(self.commander_buffer)
        else:
            policy_loss, value_loss, entropy = 0.0, 0.0, 0.0

        # Get SNN metrics
        snn_sparsity = {}
        if hasattr(self.interceptors.actor, "snn"):
            snn_sparsity = self.interceptors.actor.snn.get_spike_counts()

        metrics = {
            "commander_reward": cmd_reward,
            "interceptor_reward": int_reward,
            "mean_reward": (cmd_reward + int_reward) / 2.0,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "global_step": self.global_step,
            "episode": self.episode,
            "curriculum_phase": curr_config.get("curriculum_phase", 1),
            "snn_sparsity": snn_sparsity,
        }

        self.episode += 1
        self._episode_rewards.append(metrics["mean_reward"])
        return metrics

    def run(self, max_episodes: Optional[int] = None) -> None:
        """Run the full training loop."""
        self._init_env()
        target_episodes = max_episodes or (self.total_timesteps // self.rollout_steps)

        log.info(f"Starting MARL training for {target_episodes} episodes")

        for ep in range(target_episodes):
            metrics = self.train_step()

            # Callbacks
            self.checkpoint_cb.on_episode_end(self.episode, metrics)
            self.wandb_cb.on_episode_end(self.episode, metrics)

            if self.curriculum.is_phase_transition(self.global_step):
                log.info(f"Curriculum phase transition at step {self.global_step}")
                self._init_env()

            if ep % 10 == 0:
                log.info(
                    f"Episode {self.episode} | Step {self.global_step} | "
                    f"Cmd Reward: {metrics['commander_reward']:.2f} | "
                    f"Int Reward: {metrics['interceptor_reward']:.2f} | "
                    f"Phase: {metrics['curriculum_phase']}"
                )

        log.info("Training complete.")


if __name__ == "__main__":
    trainer = MARLTrainer(
        config={
            "marl": {"total_timesteps": 100, "rollout_steps": 10, "batch_size": 8},
            "swarm": {},
            "snn": {},
            "use_wandb": False,
            "checkpoint_dir": "/tmp/test_checkpoints",
        }
    )
    trainer._init_env()
    metrics = trainer.train_step()
    print(f"Trainer step metrics: {list(metrics.keys())}")
    print("trainer.py OK")
