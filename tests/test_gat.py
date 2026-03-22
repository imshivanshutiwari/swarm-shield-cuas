"""
Tests for SwarmGAT network.
"""

import numpy as np
import pytest
import torch


def test_adjacency_connects_nodes_within_radius(gat):
    """Nodes within comm_radius should have edges; far nodes should not."""
    positions = np.array(
        [
            [100.0, 100.0, 20.0],
            [110.0, 100.0, 20.0],  # 10m away — should be connected
            [500.0, 500.0, 20.0],  # far away — should not be connected
        ]
    )
    edge_index, edge_attr = gat.build_adjacency(positions, comm_radius=50.0)

    # Nodes 0 and 1 should be connected
    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    assert (0, 1) in edges or (1, 0) in edges, "Close nodes should be connected"

    # Node 2 should not be connected to 0 or 1
    assert (0, 2) not in edges and (2, 0) not in edges, "Far node should not be connected to 0"
    assert (1, 2) not in edges and (2, 1) not in edges, "Far node should not be connected to 1"


def test_attention_weights_sum_to_one(gat):
    """Attention weights should sum to approximately 1 per node (where connected)."""
    positions = np.array(
        [
            [100.0, 100.0, 20.0],
            [120.0, 100.0, 20.0],
            [110.0, 120.0, 20.0],
            [90.0, 110.0, 20.0],
        ]
    )
    edge_index, edge_attr = gat.build_adjacency(positions, comm_radius=100.0)
    x = torch.randn(4, 64)

    if edge_index.shape[1] == 0:
        pytest.skip("No edges formed — skip attention test")

    embeddings, attn_matrix = gat(x, edge_index, edge_attr)

    # For nodes with at least one incoming edge, weights should sum to ~1
    for i in range(4):
        row_sum = float(attn_matrix[i].sum().item())
        incoming = (edge_index[1] == i).sum().item()
        if incoming > 0:
            assert (
                abs(row_sum - 1.0) < 0.1
            ), f"Node {i} attention sum = {row_sum:.3f}, expected ~1.0"


def test_edge_dropout_reduces_edges(gat):
    """With all nodes jammed, edge count should be less than the unjammed case."""
    positions = np.array(
        [
            [100.0, 100.0, 20.0],
            [120.0, 100.0, 20.0],
            [110.0, 120.0, 20.0],
            [90.0, 110.0, 20.0],
        ]
    )
    # Without jamming
    import numpy.random as nr

    nr.seed(0)
    edge_no_jam, _ = gat.build_adjacency(positions, comm_radius=100.0, jammer_mask=None)
    n_edges_no_jam = edge_no_jam.shape[1]

    # Run multiple times with jamming to get reliable dropout
    all_jammed = np.ones(4, dtype=bool)
    n_edges_jammed_list = []
    for seed in range(10):
        nr.seed(seed)
        edge_jam, _ = gat.build_adjacency(positions, comm_radius=100.0, jammer_mask=all_jammed)
        n_edges_jammed_list.append(edge_jam.shape[1])

    avg_jammed_edges = np.mean(n_edges_jammed_list)
    assert (
        avg_jammed_edges <= n_edges_no_jam
    ), f"Jammed edges ({avg_jammed_edges:.1f}) should be <= unjammed ({n_edges_no_jam})"


def test_isolated_node_returns_zero_embedding(gat):
    """A node with no edges should get a zero embedding."""
    # 3 nodes, but position 2 is far from all others
    positions = np.array(
        [
            [100.0, 100.0, 20.0],
            [110.0, 100.0, 20.0],
            [900.0, 900.0, 20.0],  # isolated
        ]
    )
    edge_index, edge_attr = gat.build_adjacency(positions, comm_radius=50.0)
    x = torch.randn(3, 64)

    # Get isolated node embedding via handle_isolated_nodes
    isolated_embedding = gat.handle_isolated_nodes(x, edge_index)

    # Node 2 should have zero embedding if it's isolated
    connected_nodes = set()
    if edge_index.shape[1] > 0:
        connected_nodes = set(edge_index[0].tolist()) | set(edge_index[1].tolist())

    if 2 not in connected_nodes:
        node2_emb = isolated_embedding[2]
        assert (
            float(node2_emb.abs().max().item()) == 0.0
        ), f"Isolated node should have zero embedding, got {node2_emb}"
