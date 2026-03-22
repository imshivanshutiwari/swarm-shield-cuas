import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_seed() -> int:
    """Generate a random seed."""
    return random.randint(0, 2**31 - 1)


if __name__ == "__main__":
    set_seed(42)
    print("Seed set successfully.")
