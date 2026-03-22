from typing import Any, Dict, List

import torch
import torch.nn as nn

from models.snn_network import SNNNetwork


class EnergyProfiler:
    """
    Profiles energy consumption of SNN vs ANN models.
    Uses spike count-based energy estimation:
      E_SNN ≈ E_AC * total_spike_ops
      E_ANN ≈ E_MAC * total_multiply_accumulate_ops
    where E_AC << E_MAC (typically E_AC = 0.9pJ, E_MAC = 4.6pJ).
    """

    E_MAC_PJ = 4.6  # energy per MAC op in picojoules
    E_AC_PJ = 0.9  # energy per AC op in picojoules

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._mac_ops: int = 0
        self._spike_ops: int = 0
        self._hooks: List[Any] = []

    def profile_ann_energy(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Profile energy consumption of ANN forward pass.
        Returns energy in picojoules and number of MAC operations.
        """
        self._mac_ops = 0
        hooks = []

        def mac_counter(module, inp, output):
            if isinstance(module, nn.Linear):
                b = inp[0].shape[0] if inp[0].dim() > 1 else 1
                macs = int(b * module.weight.shape[0] * module.weight.shape[1])
                self._mac_ops += macs

        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(mac_counter))

        with torch.no_grad():
            self.model(x)

        for h in hooks:
            h.remove()

        energy_pj = self._mac_ops * self.E_MAC_PJ
        return {
            "mac_ops": self._mac_ops,
            "energy_pj": energy_pj,
            "energy_uj": energy_pj * 1e-6,
        }

    def profile_snn_energy(
        self,
        snn_model: SNNNetwork,
        x: torch.Tensor,
        n_timesteps: int = 8,
    ) -> Dict[str, float]:
        """
        Profile energy consumption of SNN forward pass using spike counts.
        """
        with torch.no_grad():
            _ = snn_model(x)
            spike_counts = snn_model.get_spike_counts()

        # Estimate total AC operations = sum(spike_rate * layer_size * fan_out)
        layer_sizes = {
            "lif1": 256,
            "lif2": 128,
            "lif3": 64,
        }
        layer_fanout = {
            "lif1": 128,  # 256 -> 128 Linear
            "lif2": 64,  # 128 -> 64 Linear
            "lif3": snn_model.output_dim,
        }

        total_ac_ops = 0
        spike_details: Dict[str, int] = {}
        for layer_name, spike_rate in spike_counts.items():
            n_neurons = layer_sizes.get(layer_name, 64)
            fanout = layer_fanout.get(layer_name, 64)
            batch_size = x.shape[0]
            # Total spikes = rate * neurons * batch * timesteps
            n_spikes = int(spike_rate * n_neurons * batch_size * n_timesteps)
            ac_ops = n_spikes * fanout
            total_ac_ops += ac_ops
            spike_details[layer_name] = n_spikes

        energy_pj = total_ac_ops * self.E_AC_PJ

        return {
            "ac_ops": total_ac_ops,
            "energy_pj": energy_pj,
            "energy_uj": energy_pj * 1e-6,
            "spike_counts": spike_details,
        }

    def compute_energy_ratio(
        self,
        ann_energy: Dict[str, float],
        snn_energy: Dict[str, float],
    ) -> float:
        """
        Compute SNN / ANN energy ratio.
        Values < 1.0 mean SNN is more energy-efficient.
        """
        ann_e = ann_energy.get("energy_pj", 1.0)
        snn_e = snn_energy.get("energy_pj", 1.0)
        return float(snn_e / max(ann_e, 1e-12))

    def generate_report(
        self,
        ann_energy: Dict[str, float],
        snn_energy: Dict[str, float],
    ) -> str:
        """Generate a human-readable energy comparison report."""
        ratio = self.compute_energy_ratio(ann_energy, snn_energy)
        report = (
            f"=== Energy Profiling Report ===\n"
            f"ANN Energy:  {ann_energy['energy_pj']:.2f} pJ "
            f"({ann_energy['mac_ops']:,} MAC ops)\n"
            f"SNN Energy:  {snn_energy['energy_pj']:.2f} pJ "
            f"({snn_energy['ac_ops']:,} AC ops)\n"
            f"SNN/ANN Ratio: {ratio:.4f}\n"
            f"Energy Saving: {(1.0 - ratio) * 100:.1f}%\n"
        )
        return report


if __name__ == "__main__":
    # Build equivalent ANN
    ann = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
    )
    snn = SNNNetwork(input_dim=128, output_dim=32)
    profiler = EnergyProfiler(ann)

    x = torch.randn(4, 128)
    ann_energy = profiler.profile_ann_energy(x)
    snn_energy = profiler.profile_snn_energy(snn, x)
    print(profiler.generate_report(ann_energy, snn_energy))
    print("energy_profiler.py OK")
