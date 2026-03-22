"""
Engagement visualization: plots reward curves, engagement logs, and statistics.
"""

import os
import sys
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EngagementViz:
    """
    Visualizes engagement statistics and reward curves.
    """

    def __init__(self) -> None:
        self._figures: List[plt.Figure] = []

    def plot_reward_curve(
        self,
        rewards: List[float],
        label: str = "Reward",
        smoothing_window: int = 20,
        title: str = "Training Reward Curve",
    ) -> plt.Figure:
        """Plot smoothed reward curve over episodes."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rewards, alpha=0.3, color="blue", label="Raw")
        if len(rewards) >= smoothing_window:
            smoothed = np.convolve(
                rewards, np.ones(smoothing_window) / smoothing_window, mode="valid"
            )
            ax.plot(
                range(smoothing_window - 1, len(rewards)),
                smoothed,
                color="blue",
                label=f"Smoothed (w={smoothing_window})",
                linewidth=2,
            )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._figures.append(fig)
        return fig

    def plot_curriculum_overlay(
        self,
        rewards: List[float],
        phase_boundaries: Optional[List[int]] = None,
    ) -> plt.Figure:
        """Plot reward curve with curriculum phase overlays."""
        if phase_boundaries is None:
            phase_boundaries = [len(rewards) // 3, 2 * len(rewards) // 3]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(rewards, color="blue", alpha=0.7, label="Reward")

        phase_colors = ["#90EE90", "#FFD700", "#FF6347"]
        phase_names = ["Phase 1: Easy", "Phase 2: Medium", "Phase 3: Hard"]
        boundaries = [0] + phase_boundaries + [len(rewards)]

        for i in range(len(phase_names)):
            start = boundaries[i]
            end = boundaries[min(i + 1, len(boundaries) - 1)]
            ax.axvspan(start, end, alpha=0.15, color=phase_colors[i], label=phase_names[i])

        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Training Reward with Curriculum Phases")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._figures.append(fig)
        return fig

    def plot_engagement_log(
        self, engagement_log: List[Dict[str, Any]], n_episodes: int = 10
    ) -> plt.Figure:
        """Plot engagement timeline: when each target was neutralized."""
        fig, ax = plt.subplots(figsize=(10, 5))
        if not engagement_log:
            ax.text(0.5, 0.5, "No engagements", ha="center", va="center")
            return fig

        steps = [e.get("step", 0) for e in engagement_log]
        targets = [e.get("target", 0) for e in engagement_log]
        ax.scatter(steps, targets, c="green", s=100, marker="x", label="Neutralized")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Target ID")
        ax.set_title("Engagement Timeline")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._figures.append(fig)
        return fig

    def plot_multi_agent_rewards(
        self,
        commander_rewards: List[float],
        interceptor_rewards: List[float],
    ) -> plt.Figure:
        """Plot commander and interceptor rewards on the same axes."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(commander_rewards, label="Commander (MAPPO)", color="blue", linewidth=1.5)
        ax.plot(interceptor_rewards, label="Interceptors (MADDPG)", color="green", linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Commander vs Interceptor Rewards")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._figures.append(fig)
        return fig

    def plot_neutralization_rate(self, rates: List[float]) -> plt.Figure:
        """Plot neutralization rate over episodes."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rates, color="green", linewidth=1.5)
        ax.axhline(y=80.0, color="red", linestyle="--", label="80% target")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Neutralization Rate (%)")
        ax.set_title("Swarm Neutralization Rate")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._figures.append(fig)
        return fig

    def save_all(self, output_dir: str = "assets/results") -> None:
        """Save all figures to the output directory."""
        os.makedirs(output_dir, exist_ok=True)
        for i, fig in enumerate(self._figures):
            path = os.path.join(output_dir, f"engagement_viz_{i}.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close("all")


if __name__ == "__main__":
    viz = EngagementViz()
    rewards = list(np.random.randn(100).cumsum())
    fig1 = viz.plot_reward_curve(rewards)
    fig2 = viz.plot_neutralization_rate([50 + i * 0.3 for i in range(100)])
    print(f"Created {len(viz._figures)} figures")
    plt.close("all")
    print("engagement_viz.py OK")
