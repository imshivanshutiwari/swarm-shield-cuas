"""
Attention visualization for GAT communication graphs.
"""

import os
import sys
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AttentionViz:
    """
    Visualizes attention weights from the GAT communication network.
    Shows agent-to-agent attention as edge opacity.
    """

    def __init__(self) -> None:
        self._figures: List[plt.Figure] = []

    def plot_attention_matrix(
        self,
        attn_matrix: np.ndarray,
        agent_labels: Optional[List[str]] = None,
        title: str = "GAT Attention Matrix",
    ) -> plt.Figure:
        """Plot the N x N attention matrix as a heatmap."""
        n = attn_matrix.shape[0]
        if agent_labels is None:
            agent_labels = [f"Agent {i}" for i in range(n)]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(attn_matrix, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(agent_labels, rotation=45, ha="right")
        ax.set_yticklabels(agent_labels)
        ax.set_title(title)
        ax.set_xlabel("Source Agent")
        ax.set_ylabel("Target Agent")
        plt.colorbar(im, ax=ax, label="Attention Weight")

        # Add value annotations
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    f"{attn_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if attn_matrix[i, j] < 0.5 else "white",
                )

        plt.tight_layout()
        self._figures.append(fig)
        return fig

    def plot_attention_on_map(
        self,
        positions: np.ndarray,
        attn_matrix: np.ndarray,
        grid_size: int = 500,
        jammer_positions: Optional[List[np.ndarray]] = None,
        title: str = "GAT Attention Edges on Map",
    ) -> plt.Figure:
        """
        Plot agent positions with attention-weighted edges on the 2D map.
        Edge opacity = attention weight.
        """
        n = positions.shape[0]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color="white")

        # Draw edges with attention weights
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                alpha = float(attn_matrix[i, j])
                if alpha > 0.05:
                    ax.plot(
                        [positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]],
                        color="purple",
                        alpha=min(alpha, 1.0),
                        linewidth=alpha * 3,
                    )

        # Draw agents
        colors = plt.cm.get_cmap("rainbow")(np.linspace(0, 1, n))
        for i in range(n):
            ax.scatter(positions[i, 0], positions[i, 1], c=[colors[i]], s=150, marker="^", zorder=5)
            ax.annotate(
                f"A{i}",
                (positions[i, 0], positions[i, 1] + 8),
                ha="center",
                color="white",
                fontsize=9,
            )

        # Draw jammer zones
        if jammer_positions:
            import matplotlib.patches as patches

            for jp in jammer_positions:
                circle = patches.Circle(jp[:2], 80, color="red", alpha=0.2, fill=True)
                ax.add_patch(circle)

        ax.tick_params(colors="gray")
        plt.tight_layout()
        self._figures.append(fig)
        return fig

    def plot_attention_entropy(
        self,
        entropy_history: List[float],
        title: str = "GAT Attention Entropy Over Training",
    ) -> plt.Figure:
        """Plot attention entropy over training to track communication diversity."""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(entropy_history, color="purple", linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Attention Entropy (bits)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._figures.append(fig)
        return fig

    def compute_attention_entropy(self, attn_matrix: np.ndarray) -> float:
        """Compute entropy of attention distribution (measures communication diversity)."""
        n = attn_matrix.shape[0]
        total_entropy = 0.0
        for i in range(n):
            row = attn_matrix[i]
            row_sum = row.sum()
            if row_sum < 1e-8:
                continue
            p = row / row_sum
            p = p[p > 1e-12]
            total_entropy += float(-np.sum(p * np.log2(p + 1e-12)))
        return total_entropy / max(n, 1)

    def save_all(self, output_dir: str = "assets/results") -> None:
        """Save all figures."""
        os.makedirs(output_dir, exist_ok=True)
        for i, fig in enumerate(self._figures):
            path = os.path.join(output_dir, f"attention_viz_{i}.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close("all")


if __name__ == "__main__":
    viz = AttentionViz()
    attn = np.random.rand(4, 4)
    attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)

    fig1 = viz.plot_attention_matrix(attn)
    positions = np.array([[100, 100], [200, 150], [300, 200], [150, 300]], dtype=np.float32)
    fig2 = viz.plot_attention_on_map(positions, attn)
    entropy = viz.compute_attention_entropy(attn)
    print(f"Attention entropy: {entropy:.3f} bits")
    plt.close("all")
    print("attention_viz.py OK")
