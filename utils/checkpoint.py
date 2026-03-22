import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    state: Dict[str, Any],
    path: str,
    filename: str = "checkpoint.pt",
) -> str:
    """Save a training checkpoint to disk."""
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    torch.save(state, full_path)
    return full_path


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint from disk.

    weights_only=False is required because checkpoints contain numpy arrays inside
    the metrics dict saved by CheckpointCallback. Only load checkpoints produced by
    this project's own training pipeline.
    """
    return torch.load(path, map_location=map_location, weights_only=False)  # nosec


def save_model(model: torch.nn.Module, path: str) -> None:
    """Save model state dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str, map_location: str = "cpu") -> torch.nn.Module:
    """Load model state dict into a model instance."""
    state_dict = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict)
    return model


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the path to the most recent checkpoint file in the directory."""
    checkpoint_dir_path = Path(checkpoint_dir)
    if not checkpoint_dir_path.exists():
        return None
    checkpoints = sorted(checkpoint_dir_path.glob("*.pt"), key=os.path.getmtime)
    if checkpoints:
        return str(checkpoints[-1])
    return None


def build_checkpoint_name(episode: int, reward: float) -> str:
    """Build a standardized checkpoint filename."""
    ts = int(time.time())
    return f"ckpt_ep{episode:06d}_r{reward:.2f}_{ts}.pt"


if __name__ == "__main__":
    print("Checkpoint utilities loaded successfully.")
