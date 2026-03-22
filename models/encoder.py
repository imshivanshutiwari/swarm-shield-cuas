import torch
import torch.nn as nn


class ObservationEncoder(nn.Module):
    """
    Encodes heterogeneous observation dict into a fixed-size embedding.
    Handles radar returns, RF spectrogram, own state, and GAT neighbor obs.
    """

    def __init__(
        self,
        n_targets: int = 10,
        n_interceptors: int = 4,
        gat_dim: int = 64,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_targets = n_targets
        self.n_interceptors = n_interceptors
        self.gat_dim = gat_dim
        self.output_dim = output_dim

        # Radar returns encoder: (n_targets, 4) -> 64
        radar_input_dim = n_targets * 4
        self.radar_encoder = nn.Sequential(
            nn.Linear(radar_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # RF spectrogram encoder: (64, 64) -> 64
        self.rf_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        # GAT neighbor obs encoder: ((n_interceptors-1) * gat_dim) -> 64
        gat_input_dim = (n_interceptors - 1) * gat_dim
        self.gat_encoder = nn.Sequential(
            nn.Linear(gat_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Own state encoder: 6 -> 32
        self.own_state_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
        )

        # Commander task embedding
        self.task_embedding = nn.Embedding(n_targets + 1, 16)

        # Fusion layer: 64+64+64+32+16 -> output_dim
        fusion_input = 64 + 64 + 64 + 32 + 16
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        radar_returns: torch.Tensor,
        rf_spectrogram: torch.Tensor,
        gat_neighbor_obs: torch.Tensor,
        own_state: torch.Tensor,
        commander_task: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            radar_returns: (B, n_targets, 4)
            rf_spectrogram: (B, 64, 64)
            gat_neighbor_obs: (B, n_interceptors-1, 64)
            own_state: (B, 6)
            commander_task: (B,) int64
        Returns:
            embedding: (B, output_dim)
        """
        B = radar_returns.shape[0]

        radar_enc = self.radar_encoder(radar_returns.view(B, -1))
        rf_enc = self.rf_encoder(rf_spectrogram)
        gat_enc = self.gat_encoder(gat_neighbor_obs.view(B, -1))
        own_enc = self.own_state_encoder(own_state)
        task_enc = self.task_embedding(commander_task)

        fused = torch.cat([radar_enc, rf_enc, gat_enc, own_enc, task_enc], dim=-1)
        return self.fusion(fused)


class StateEncoder(nn.Module):
    """Simple MLP encoder for flat state vectors."""

    def __init__(self, input_dim: int, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    enc = ObservationEncoder()
    B = 2
    radar = torch.randn(B, 10, 4)
    rf = torch.randn(B, 64, 64)
    gat = torch.randn(B, 3, 64)
    own = torch.randn(B, 6)
    task = torch.zeros(B, dtype=torch.long)
    out = enc(radar, rf, gat, own, task)
    print(f"Encoder output shape: {out.shape}")
    print("encoder.py OK")
