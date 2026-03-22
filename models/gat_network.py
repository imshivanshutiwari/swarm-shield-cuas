from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

try:
    from torch_geometric.nn import GATv2Conv
except ImportError:
    raise ImportError("torch_geometric is required for SwarmGAT")


class SwarmGAT(nn.Module):
    """
    Graph Attention Network for swarm communication.
    Uses GATv2Conv layers with edge features including jamming simulation.
    Edge features: [relative_bearing, range, range_rate, link_quality, is_jammed]
    """

    def __init__(
        self,
        node_feat_dim: int = 64,
        edge_feat_dim: int = 5,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout

        # Layer 1: concat output -> heads * hidden_dim
        self.conv1 = GATv2Conv(
            node_feat_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_feat_dim,
            concat=True,
        )
        self.elu1 = nn.ELU()

        # Layer 2: average over heads -> hidden_dim
        self.conv2 = GATv2Conv(
            hidden_dim * heads,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_feat_dim,
            concat=False,
        )

        # Store attention weights from last forward
        self._last_attn: Optional[torch.Tensor] = None
        self._last_n_nodes: int = 0

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N, node_feat_dim) node features
            edge_index: (2, E) edge connectivity
            edge_attr: (E, edge_feat_dim) edge features
        Returns:
            embeddings: (N, hidden_dim)
            attention_weights: (N, N) aggregated attention (mean over heads)
        """
        n_nodes = x.shape[0]
        self._last_n_nodes = n_nodes

        # Handle isolated nodes
        if edge_index.shape[1] == 0:
            embeddings = torch.zeros(n_nodes, self.hidden_dim, device=x.device)
            attn_matrix = torch.zeros(n_nodes, n_nodes, device=x.device)
            self._last_attn = attn_matrix
            return embeddings, attn_matrix

        # Conv1: returns (output, (edge_index_out, alpha))
        conv1_out = self.conv1(x, edge_index, edge_attr, return_attention_weights=True)
        h1, (_, alpha1) = conv1_out
        h1 = self.elu1(h1)

        # Conv2: returns (output, (edge_index_out, alpha))
        conv2_out = self.conv2(h1, edge_index, edge_attr, return_attention_weights=True)
        h2, (ei2, alpha2) = conv2_out

        # Build attention matrix (N x N)
        attn_matrix = self._build_attention_matrix(ei2, alpha2, n_nodes)
        self._last_attn = attn_matrix

        return h2, attn_matrix

    def _build_attention_matrix(
        self,
        edge_index_out: torch.Tensor,
        alpha: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """Build dense N x N attention matrix from sparse attention weights."""
        if alpha.dim() == 2:
            alpha = alpha.mean(dim=-1)  # (E,) mean over heads

        attn_matrix = torch.zeros(n_nodes, n_nodes, device=alpha.device)
        src, dst = edge_index_out[0], edge_index_out[1]
        attn_matrix[dst, src] = alpha

        # Row-normalize so weights sum to ~1 per node (where connected)
        row_sum = attn_matrix.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        # Only normalize rows that have at least one non-zero
        has_edges = (attn_matrix.sum(dim=-1) > 0).unsqueeze(-1)
        normalized = torch.where(has_edges, attn_matrix / row_sum, attn_matrix)
        return normalized

    def build_adjacency(
        self,
        positions: np.ndarray,
        comm_radius: float = 150.0,
        jammer_mask: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edge_index and edge_attr for a set of agent positions.
        Applies edge dropout for jammed nodes.

        Args:
            positions: (N, 3) array of 3D positions
            comm_radius: communication radius in meters
            jammer_mask: (N,) boolean array, True = node is jammed
        Returns:
            edge_index: (2, E) tensor
            edge_attr: (E, 5) tensor [bearing, range, range_rate, quality, is_jammed]
        """
        n = len(positions)
        if jammer_mask is None:
            jammer_mask = np.zeros(n, dtype=bool)

        edges_src, edges_dst = [], []
        edge_feats = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = positions[j] - positions[i]
                dist = float(np.linalg.norm(diff))
                if dist > comm_radius:
                    continue

                # Apply dropout on jammed edges (50% drop probability)
                if jammer_mask[i] or jammer_mask[j]:
                    if np.random.rand() < 0.5:
                        continue

                bearing = float(np.arctan2(diff[1], diff[0]))
                range_ = dist
                range_rate = 0.0  # simplified: no velocity information
                link_quality = float(np.clip(1.0 - dist / comm_radius, 0.0, 1.0))
                is_jammed = float(jammer_mask[i] or jammer_mask[j])

                edges_src.append(i)
                edges_dst.append(j)
                edge_feats.append([bearing, range_, range_rate, link_quality, is_jammed])

        if not edges_src:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 5), dtype=torch.float32)
        else:
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
            edge_attr = torch.tensor(edge_feats, dtype=torch.float32)

        return edge_index, edge_attr

    def get_attention_matrix(self) -> torch.Tensor:
        """Return the last computed attention matrix."""
        if self._last_attn is None:
            return torch.zeros(1, 1)
        return self._last_attn

    def handle_isolated_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return zero embedding for nodes with no edges."""
        n_nodes = x.shape[0]
        if edge_index.shape[1] == 0:
            return torch.zeros(n_nodes, self.hidden_dim, device=x.device)
        connected_nodes = set(edge_index[0].tolist()) | set(edge_index[1].tolist())
        embeddings = torch.zeros(n_nodes, self.hidden_dim, device=x.device)
        for node_idx in range(n_nodes):
            if node_idx not in connected_nodes:
                embeddings[node_idx] = 0.0
        return embeddings


if __name__ == "__main__":
    gat = SwarmGAT(node_feat_dim=64)
    positions = np.array([[100.0, 100.0, 20.0], [110.0, 100.0, 20.0], [500.0, 500.0, 20.0]])
    edge_index, edge_attr = gat.build_adjacency(positions, comm_radius=150.0)
    x = torch.randn(3, 64)
    embeddings, attn = gat(x, edge_index, edge_attr)
    print(f"GAT embeddings shape: {embeddings.shape}")
    print(f"Attention matrix shape: {attn.shape}")
    print("gat_network.py OK")
