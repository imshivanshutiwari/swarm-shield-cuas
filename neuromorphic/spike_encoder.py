import numpy as np
import torch


class SpikeEncoder:
    """
    Encodes raw sensor data into spike trains for neuromorphic processing.
    Supports rate coding and temporal coding.
    """

    def __init__(
        self,
        n_timesteps: int = 8,
        encoding: str = "rate",
        threshold: float = 0.5,
    ) -> None:
        assert encoding in ("rate", "temporal"), "encoding must be 'rate' or 'temporal'"
        self.n_timesteps = n_timesteps
        self.encoding = encoding
        self.threshold = threshold

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input tensor into spike train.
        Args:
            x: (B, D) input tensor (values in any range)
        Returns:
            spikes: (T, B, D) binary spike tensor
        """
        if self.encoding == "rate":
            return self._rate_encode(x)
        else:
            return self._temporal_encode(x)

    def _rate_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Poisson rate coding: spike probability proportional to input magnitude.
        Input is normalized to [0, 1] before encoding.
        """
        x_norm = torch.sigmoid(x)  # map to (0, 1)
        spikes = torch.stack(
            [torch.bernoulli(x_norm) for _ in range(self.n_timesteps)],
            dim=0,
        )
        return spikes  # (T, B, D)

    def _temporal_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Temporal coding: higher values fire earlier.
        Neuron i fires at timestep t = T - round(x_norm[i] * T).
        """
        x_norm = torch.sigmoid(x)  # (B, D)
        fire_time = (self.n_timesteps - 1) - (x_norm * (self.n_timesteps - 1)).long()
        T = self.n_timesteps
        spikes = torch.zeros(T, *x.shape, dtype=torch.float32)
        for t in range(T):
            spikes[t] = (fire_time == t).float()
        return spikes  # (T, B, D)

    def decode_rate(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Decode spike train back to rate estimate.
        Args:
            spikes: (T, B, D)
        Returns:
            rates: (B, D) firing rates in [0, 1]
        """
        return spikes.float().mean(dim=0)

    def encode_radar_return(self, radar_data: np.ndarray) -> torch.Tensor:
        """
        Encode radar return (range, bearing, doppler, SNR) into spike train.
        Args:
            radar_data: (n_targets, 4) numpy array
        Returns:
            spikes: (T, n_targets, 4) spike train
        """
        x = torch.tensor(radar_data, dtype=torch.float32)
        # Normalize each channel separately
        x_flat = x.view(1, -1)
        return self.encode(x_flat).squeeze(1).view(self.n_timesteps, *radar_data.shape)

    def encode_rf_spectrogram(self, spectrogram: np.ndarray) -> torch.Tensor:
        """
        Encode RF spectrogram (64x64) into spike train.
        Args:
            spectrogram: (64, 64) numpy array in [0, 1]
        Returns:
            spikes: (T, 64, 64) spike train
        """
        x = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
        encoded = self.encode(x)  # (T, 1, 64*64) after flatten
        return encoded.squeeze(1).view(self.n_timesteps, 64, 64)


if __name__ == "__main__":
    encoder = SpikeEncoder(n_timesteps=8, encoding="rate")
    x = torch.randn(4, 128)
    spikes = encoder.encode(x)
    print(f"Spike train shape: {spikes.shape}")
    rates = encoder.decode_rate(spikes)
    print(f"Decoded rates shape: {rates.shape}")
    print("spike_encoder.py OK")
