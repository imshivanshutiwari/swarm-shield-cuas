"""
Real-time animated swarm visualization using matplotlib.
"""

import io
import os
import sys
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.cuas_env import CUASEnv  # noqa: E402


class SwarmRenderer:
    """
    Real-time animated matplotlib visualization of the C-UAS environment.
    """

    TYPE_COLORS = {"kamikaze": "red", "isr": "orange", "jammer": "purple"}

    def __init__(self, env: Optional[CUASEnv] = None) -> None:
        self.env = env
        self._fig: Optional[plt.Figure] = None
        self._frames: List[np.ndarray] = []

    def render_frame(self, env_state: Optional[Dict[str, Any]] = None) -> plt.Figure:
        """
        Render the current environment state as a matplotlib figure.
        Shows: terrain, enemy drones, interceptors, engagement lines,
               jammer zones, GPS denial zones, radar arcs, attention edges,
               score panel.
        """
        env = self.env
        if env is None and env_state is not None:
            # Use provided state dict for rendering
            return self._render_from_state(env_state)
        if env is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, "No environment", ha="center", va="center")
            return fig

        fig = env.render(mode="human")
        self._frames.append(self._fig_to_array(fig))
        return fig

    def _render_from_state(self, state: Dict[str, Any]) -> plt.Figure:
        """Render from a state dictionary."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        grid_size = state.get("grid_size", 500)
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_facecolor("#1a1a2e")
        ax.set_title("SWARM-SHIELD: C-UAS State", color="white")

        # Enemy drones
        for drone_info in state.get("enemy_drones", []):
            color = self.TYPE_COLORS.get(drone_info.get("type", "kamikaze"), "red")
            pos = drone_info.get("position", [0, 0])
            alpha = 1.0 if drone_info.get("alive", True) else 0.3
            ax.scatter(pos[0], pos[1], c=color, s=80, alpha=alpha)

        # Interceptors
        for int_info in state.get("interceptors", []):
            pos = int_info.get("position", [0, 0])
            ax.scatter(pos[0], pos[1], c="cyan", s=120, marker="^")

        plt.tight_layout()
        return fig

    def _fig_to_array(self, fig: plt.Figure) -> np.ndarray:
        """Convert matplotlib figure to numpy RGB array."""
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        buf.seek(0)
        import PIL.Image

        img = PIL.Image.open(buf)
        arr = np.array(img)[:, :, :3]
        buf.close()
        return arr

    def create_episode_gif(self, frames: List[np.ndarray], path: str) -> None:
        """Save list of RGB frames as an animated GIF."""
        import PIL.Image

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        pil_frames = [PIL.Image.fromarray(f.astype(np.uint8)) for f in frames]
        if pil_frames:
            pil_frames[0].save(
                path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=100,
                loop=0,
            )

    def export_frame(self, path: str) -> None:
        """Export the last rendered frame to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        if self._frames:
            import PIL.Image

            PIL.Image.fromarray(self._frames[-1].astype(np.uint8)).save(path)


if __name__ == "__main__":
    env = CUASEnv()
    env.reset(seed=42)
    renderer = SwarmRenderer(env=env)
    fig = renderer.render_frame()
    print(f"Rendered frame: {fig}")
    plt.close("all")
    print("swarm_renderer.py OK")
