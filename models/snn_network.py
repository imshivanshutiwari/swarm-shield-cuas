from typing import Dict

import torch
import torch.nn as nn
import numpy as np

from spikingjelly.activation_based import neuron, functional, surrogate


class SNNNetwork(nn.Module):
    """
    Spiking Neural Network using SpikingJelly's activation-based framework.
    Architecture: Linear -> LIF -> Linear -> LIF -> Linear -> LIF -> Linear
    Uses multi-step mode with T=8 timesteps and ATan surrogate gradient.
    """

    T: int = 8  # number of timesteps

    def __init__(self, input_dim: int = 128, output_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        surrogate_fn = surrogate.ATan()

        self.fc1 = nn.Linear(input_dim, 256)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate_fn, detach_reset=True)
        self.fc2 = nn.Linear(256, 128)
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate_fn, detach_reset=True)
        self.fc3 = nn.Linear(128, 64)
        self.lif3 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate_fn, detach_reset=True)
        self.fc4 = nn.Linear(64, output_dim)

        # Set multi-step mode
        functional.set_step_mode(self, step_mode="m")

        # Set backend
        backend = "cupy" if torch.cuda.is_available() else "torch"
        functional.set_backend(self, backend=backend)

        # Track spike counts per layer (populated during forward)
        self._spike_counts: Dict[str, float] = {}
        self._total_neurons: Dict[str, int] = {}

    def encode_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input tensor into spike train via Poisson rate coding.
        Returns: (T, B, input_dim) spike tensor.
        """
        # Normalize input to [0, 1]
        x_norm = torch.sigmoid(x)
        # Poisson sampling: spike if random < rate
        spikes = torch.stack(
            [torch.bernoulli(x_norm) for _ in range(self.T)],
            dim=0,
        )
        return spikes  # (T, B, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with T timesteps.
        Args:
            x: (B, input_dim)
        Returns:
            output averaged over T timesteps: (B, output_dim)
        """
        functional.reset_net(self)

        # Encode to spike train: (T, B, input_dim)
        x_spikes = self.encode_input(x)

        # Pass through layers in multi-step mode
        # fc layers accept (T, B, *) in multi-step mode
        out1 = self.fc1(x_spikes)  # (T, B, 256)
        spk1 = self.lif1(out1)  # (T, B, 256)

        out2 = self.fc2(spk1)  # (T, B, 128)
        spk2 = self.lif2(out2)  # (T, B, 128)

        out3 = self.fc3(spk2)  # (T, B, 64)
        spk3 = self.lif3(out3)  # (T, B, 64)

        out4 = self.fc4(spk3)  # (T, B, output_dim)

        # Record spike counts
        self._spike_counts = {
            "lif1": float(spk1.mean().item()),
            "lif2": float(spk2.mean().item()),
            "lif3": float(spk3.mean().item()),
        }
        self._total_neurons = {
            "lif1": 256,
            "lif2": 128,
            "lif3": 64,
        }

        # Average over T timesteps
        return out4.mean(dim=0)  # (B, output_dim)

    def get_spike_counts(self) -> Dict[str, float]:
        """Return spike rate per LIF layer from last forward pass."""
        return dict(self._spike_counts)

    def compute_sparsity(self) -> float:
        """
        Compute spike sparsity (fraction of neurons NOT spiking).
        Target: > 0.85 (i.e., 85% of neurons silent).
        """
        if not self._spike_counts:
            return 1.0
        rates = list(self._spike_counts.values())
        mean_rate = np.mean(rates)
        return float(1.0 - mean_rate)

    def reset_states(self) -> None:
        """Clear membrane potentials of all LIF nodes."""
        functional.reset_net(self)


if __name__ == "__main__":
    model = SNNNetwork(input_dim=128, output_dim=64)
    x = torch.randn(4, 128)
    out = model(x)
    print(f"SNN output shape: {out.shape}")
    print(f"Spike counts: {model.get_spike_counts()}")
    print(f"Sparsity: {model.compute_sparsity():.3f}")
    print("snn_network.py OK")
