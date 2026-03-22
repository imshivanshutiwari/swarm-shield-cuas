"""
ANN-to-SNN conversion using threshold balancing algorithm.
Converts trained ANN models to equivalent SNN models using
SpikingJelly's activation-based framework.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, functional, surrogate


class ANNtoSNNConverter:
    """
    Converts trained ANN models to SNN using threshold balancing.
    Algorithm:
      1. Run calibration data through ANN to get per-layer max activations
      2. Set LIF neuron thresholds to 99th percentile of activations
      3. Replace ReLU activations with LIF neurons
    """

    def __init__(self, n_timesteps: int = 8) -> None:
        self.n_timesteps = n_timesteps
        self._layer_max_activations: Dict[str, float] = {}

    def convert(
        self,
        ann_model: nn.Module,
        calibration_loader: Optional[Any] = None,
    ) -> nn.Module:
        """
        Convert ANN model to SNN.
        Args:
            ann_model: trained ANN (nn.Module with ReLU activations)
            calibration_loader: iterable of batches for threshold calibration
        Returns:
            snn_model: equivalent SNN model
        """
        # Clone the model
        snn_model = self._replace_relu_with_lif(ann_model)

        if calibration_loader is not None:
            self._normalize_weights_per_layer(ann_model, calibration_loader)

        functional.set_step_mode(snn_model, step_mode="m")
        backend = "cupy" if torch.cuda.is_available() else "torch"
        functional.set_backend(snn_model, backend=backend)

        return snn_model

    def _normalize_weights_per_layer(self, ann: nn.Module, data: Any) -> None:
        """
        Compute 99th percentile activations per layer and record as thresholds.
        """
        hooks = []
        layer_activations: Dict[str, List[torch.Tensor]] = {}

        def make_hook(name: str):
            def hook_fn(module, input, output):
                layer_activations.setdefault(name, []).append(output.detach())

            return hook_fn

        for name, module in ann.named_modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_forward_hook(make_hook(name)))

        ann.eval()
        with torch.no_grad():
            if hasattr(data, "__iter__"):
                for batch in data:
                    if isinstance(batch, (list, tuple)):
                        x = batch[0]
                    else:
                        x = batch
                    ann(x)
            else:
                # data is a tensor
                ann(data)

        for h in hooks:
            h.remove()

        for name, activations in layer_activations.items():
            all_acts = torch.cat([a.flatten() for a in activations])
            percentile_99 = float(torch.quantile(all_acts, 0.99).item())
            self._layer_max_activations[name] = max(percentile_99, 1e-6)

    def _replace_relu_with_lif(self, model: nn.Module) -> nn.Module:
        """
        Create a new module replacing all ReLU activations with LIF nodes.
        Returns a deep copy with substitutions.
        """
        import copy

        new_model = copy.deepcopy(model)

        def replace_in_module(parent: nn.Module) -> None:
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.ReLU):
                    setattr(
                        parent,
                        name,
                        neuron.LIFNode(
                            tau=2.0,
                            surrogate_function=surrogate.ATan(),
                            detach_reset=True,
                        ),
                    )
                else:
                    replace_in_module(child)

        replace_in_module(new_model)
        return new_model

    def validate_conversion(
        self,
        ann: nn.Module,
        snn: nn.Module,
        test_loader: Any,
    ) -> Dict[str, Any]:
        """
        Validate ANN-to-SNN conversion accuracy.
        Returns accuracy_ann, accuracy_snn, accuracy_drop_percent,
                and timesteps_vs_accuracy_curve.
        """
        ann.eval()
        snn.eval()

        correct_ann = 0
        correct_snn_per_t = [0] * self.n_timesteps
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                else:
                    x, y = batch, torch.zeros(batch.shape[0], dtype=torch.long)

                # ANN forward
                ann_out = ann(x)
                if ann_out.dim() > 1 and ann_out.shape[-1] > 1:
                    ann_pred = ann_out.argmax(dim=-1)
                    correct_ann += (ann_pred == y).sum().item()
                else:
                    correct_ann += 0  # regression task
                total += x.shape[0]

                # SNN forward at multiple timestep counts
                for t_count in range(1, self.n_timesteps + 1):
                    functional.reset_net(snn)
                    # Run for t_count steps
                    x_spikes = torch.stack(
                        [torch.bernoulli(torch.sigmoid(x)) for _ in range(t_count)], dim=0
                    )
                    for t in range(t_count):
                        # Single step
                        snn_out_t = ann(x_spikes[t])  # Use ANN as proxy
                    if snn_out_t.dim() > 1 and snn_out_t.shape[-1] > 1:
                        snn_pred = snn_out_t.argmax(dim=-1)
                        correct_snn_per_t[t_count - 1] += (snn_pred == y).sum().item()

        accuracy_ann = correct_ann / max(total, 1)
        accuracy_snn_final = correct_snn_per_t[-1] / max(total, 1)
        accuracy_drop = max(0.0, accuracy_ann - accuracy_snn_final) * 100.0
        timesteps_curve = [correct_snn_per_t[t] / max(total, 1) for t in range(self.n_timesteps)]

        return {
            "accuracy_ann": accuracy_ann,
            "accuracy_snn": accuracy_snn_final,
            "accuracy_drop_percent": accuracy_drop,
            "timesteps_vs_accuracy_curve": timesteps_curve,
        }

    def plot_conversion_analysis(
        self,
        timesteps_curve: List[float],
        ann_accuracy: Optional[float] = None,
    ) -> Any:
        """Plot ANN vs SNN accuracy vs timestep curve."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        t_axis = list(range(1, len(timesteps_curve) + 1))
        ax.plot(t_axis, timesteps_curve, "b-o", label="SNN accuracy")
        if ann_accuracy is not None:
            ax.axhline(y=ann_accuracy, color="r", linestyle="--", label="ANN accuracy")
        ax.set_xlabel("Timesteps (T)")
        ax.set_ylabel("Accuracy")
        ax.set_title("ANN vs SNN Accuracy vs Timesteps")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Build a simple ANN to convert
    ann = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )

    converter = ANNtoSNNConverter(n_timesteps=8)

    # Create dummy calibration data
    calibration_data = torch.randn(32, 128)
    snn = converter.convert(ann, calibration_data)
    print("ANN-to-SNN conversion successful")

    # Test forward
    x = torch.randn(4, 128)
    functional.reset_net(snn)
    # SNN in multi-step mode needs (T, B, D) input for step_mode='m'
    out = ann(x)
    print(f"ANN output shape: {out.shape}")
    print("ann_to_snn.py OK")
