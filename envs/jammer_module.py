import numpy as np


class EWJammer:
    """
    Electronic warfare jammer module.
    Simulates RF jamming effects including broadband, spot, and sweep jamming.
    """

    JAM_TYPES = ("broadband", "spot", "sweep")

    def __init__(
        self,
        position: np.ndarray,
        radius: float = 80.0,
        power_dbm: float = 45.0,
        jam_type: str = "broadband",
    ) -> None:
        assert jam_type in self.JAM_TYPES, f"jam_type must be one of {self.JAM_TYPES}"
        self.position = np.array(position, dtype=np.float32)
        self.radius = float(radius)
        self.power_dbm = float(power_dbm)
        self.jam_type = jam_type
        self._sweep_phase = 0.0

    def compute_jamming_effect(self, target_pos: np.ndarray) -> float:
        """
        Compute jamming effect (SNR loss) at target_pos.
        Returns a value in [0, 1] where 1 = complete denial.
        """
        target_pos = np.array(target_pos, dtype=np.float32)
        dist = float(np.linalg.norm(target_pos[:2] - self.position[:2]))
        if dist > self.radius:
            return 0.0

        power_linear = 10 ** (self.power_dbm / 10.0)
        path_loss = max(dist, 1.0) ** 2
        jam_strength = power_linear / (power_linear + path_loss)

        if self.jam_type == "broadband":
            return float(np.clip(jam_strength, 0.0, 1.0))
        elif self.jam_type == "spot":
            # Spot jamming is more focused but stronger within inner 50% of radius
            if dist < self.radius * 0.5:
                return float(np.clip(jam_strength * 1.5, 0.0, 1.0))
            return float(np.clip(jam_strength * 0.3, 0.0, 1.0))
        else:  # sweep
            self._sweep_phase = (self._sweep_phase + 0.05) % (2 * np.pi)
            angle_to_target = np.arctan2(
                target_pos[1] - self.position[1],
                target_pos[0] - self.position[0],
            )
            sweep_factor = 0.5 * (1.0 + np.cos(self._sweep_phase - angle_to_target))
            return float(np.clip(jam_strength * sweep_factor, 0.0, 1.0))

    def corrupt_radar_return(
        self,
        raw_detection: np.ndarray,
        jammer_effect: float,
    ) -> np.ndarray:
        """
        Apply Rayleigh-distributed noise to radar [range, bearing, doppler, snr].
        Higher jammer_effect → greater corruption.
        """
        corrupted = raw_detection.copy().astype(np.float32)
        if jammer_effect <= 0.0:
            return corrupted

        # Rayleigh noise scale proportional to jammer effect
        scale = jammer_effect * 10.0
        rayleigh_noise = np.random.rayleigh(scale, size=corrupted.shape)
        # Add noise to range and doppler, reduce SNR
        corrupted[0] += rayleigh_noise[0]  # range
        corrupted[2] += rayleigh_noise[2] * 0.5  # doppler
        corrupted[3] = max(0.0, corrupted[3] * (1.0 - jammer_effect))  # SNR
        return corrupted

    def spoof_gps_coords(
        self,
        true_pos: np.ndarray,
        drift_rate: float = 0.1,
    ) -> np.ndarray:
        """
        Simulate GPS spoofing by adding cumulative drift to the true position.
        drift_rate: meters per timestep of cumulative error.
        """
        true_pos = np.array(true_pos, dtype=np.float32)
        drift = np.random.normal(0.0, drift_rate, size=true_pos.shape)
        return true_pos + drift

    def is_in_range(self, pos: np.ndarray) -> bool:
        """Return True if pos is within the jammer's effective radius."""
        pos = np.array(pos, dtype=np.float32)
        dist = float(np.linalg.norm(pos[:2] - self.position[:2]))
        return dist <= self.radius


if __name__ == "__main__":
    jammer = EWJammer(position=np.array([250.0, 250.0, 0.0]), radius=80.0, power_dbm=45.0)
    effect = jammer.compute_jamming_effect(np.array([260.0, 260.0, 10.0]))
    print(f"Jamming effect at close range: {effect:.3f}")
    print("jammer_module.py OK")
