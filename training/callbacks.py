import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from utils.logger import get_logger

log = get_logger("callbacks")


class CheckpointCallback:
    """
    Saves model checkpoints at regular intervals during training.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        save_every_n_episodes: int = 100,
        agents: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_episodes = save_every_n_episodes
        self.agents = agents or {}
        self._best_reward = -np.inf
        os.makedirs(checkpoint_dir, exist_ok=True)

    def on_episode_end(self, episode: int, metrics: Dict[str, Any]) -> None:
        """Called at the end of each episode."""
        if episode % self.save_every_n_episodes == 0:
            self._save_all(episode, metrics)

        mean_reward = metrics.get("mean_reward", 0.0)
        if mean_reward > self._best_reward:
            self._best_reward = mean_reward
            self._save_all(episode, metrics, prefix="best")

    def _save_all(self, episode: int, metrics: Dict[str, Any], prefix: str = "ckpt") -> None:
        """Save all registered agent checkpoints."""
        import torch

        for name, agent in self.agents.items():
            agent_dir = os.path.join(self.checkpoint_dir, name)
            os.makedirs(agent_dir, exist_ok=True)
            filename = f"{prefix}_ep{episode:06d}.pt"
            state: Dict[str, Any] = {"episode": episode, "metrics": metrics}

            if hasattr(agent, "actor") and hasattr(agent, "critic"):
                state["actor"] = agent.actor.state_dict()
                state["critic"] = agent.critic.state_dict()
            elif hasattr(agent, "q_network"):
                state["q_network"] = agent.q_network.state_dict()
                state["mixing_network"] = agent.mixing_network.state_dict()

            torch.save(state, os.path.join(agent_dir, filename))
            log.info(f"Saved checkpoint: {agent_dir}/{filename}")


class WandBCallback:
    """
    Logs training metrics to Weights & Biases.
    """

    def __init__(self, project: str = "swarm-shield-cuas", enabled: bool = True) -> None:
        self.project = project
        self.enabled = enabled
        self._step = 0

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to WandB."""
        if not self.enabled:
            return
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(metrics, step=step if step is not None else self._step)
        except Exception:
            pass
        self._step += 1

    def on_episode_end(self, episode: int, metrics: Dict[str, Any]) -> None:
        """Log episode metrics."""
        self.log(metrics, step=episode)


class EvaluationCallback:
    """
    Runs periodic evaluation and logs results.
    """

    def __init__(
        self,
        evaluate_fn: Callable[[], Dict[str, Any]],
        eval_every_n_episodes: int = 500,
    ) -> None:
        self.evaluate_fn = evaluate_fn
        self.eval_every_n_episodes = eval_every_n_episodes
        self._eval_results: List[Dict[str, Any]] = []

    def on_episode_end(self, episode: int, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run evaluation if it's time."""
        if episode > 0 and episode % self.eval_every_n_episodes == 0:
            results = self.evaluate_fn()
            self._eval_results.append({"episode": episode, **results})
            log.info(f"Evaluation at episode {episode}: {results}")
            return results
        return None

    def get_history(self) -> List[Dict[str, Any]]:
        """Return history of evaluation results."""
        return list(self._eval_results)


if __name__ == "__main__":
    cb = CheckpointCallback(checkpoint_dir="/tmp/test_checkpoints")
    cb.on_episode_end(100, {"mean_reward": 10.0})
    print("callbacks.py OK")
