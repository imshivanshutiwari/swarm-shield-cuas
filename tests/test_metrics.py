"""
Tests for evaluation metrics.
"""

import numpy as np

from evaluation.metrics import (
    swarm_neutralization_rate,
    ospa_distance,
    engagement_latency_ms,
)


def test_neutralization_rate_calculation():
    """neutralization_rate(8, 10) should equal 80.0."""
    rate = swarm_neutralization_rate(8, 10)
    assert rate == 80.0, f"Expected 80.0, got {rate}"

    assert swarm_neutralization_rate(0, 10) == 0.0
    assert swarm_neutralization_rate(10, 10) == 100.0
    assert swarm_neutralization_rate(0, 0) == 0.0  # edge case


def test_ospa_distance_known_case():
    """OSPA should be 0 for perfect tracking and c for completely wrong tracking."""
    # Perfect tracking
    est = np.array([[0.0, 0.0], [10.0, 0.0]])
    true_pos = np.array([[0.0, 0.0], [10.0, 0.0]])
    ospa_perfect = ospa_distance(est, true_pos, c=20.0, p=1)
    assert ospa_perfect < 1e-6, f"OSPA should be 0 for perfect tracking, got {ospa_perfect}"

    # Completely wrong tracking (one point vs empty)
    est2 = np.array([[1000.0, 1000.0]])
    true2 = np.array([[0.0, 0.0]])
    ospa_wrong = ospa_distance(est2, true2, c=20.0, p=1)
    assert (
        abs(ospa_wrong - 20.0) < 1.0
    ), f"OSPA should be ~c=20 for very wrong tracking, got {ospa_wrong}"

    # Empty vs empty
    ospa_empty = ospa_distance(np.array([]).reshape(0, 2), np.array([]).reshape(0, 2))
    assert ospa_empty == 0.0, f"OSPA should be 0 for empty sets, got {ospa_empty}"


def test_engagement_latency_under_threshold():
    """Engagement latency should be under 25ms for reasonable action timestamps."""
    # Simulate action timestamps at 10ms intervals (10 Hz)
    timestamps = [0.01 * i for i in range(10)]  # 0, 0.01, 0.02, ..., 0.09 seconds
    latency = engagement_latency_ms(timestamps)
    assert latency < 25.0, f"Latency {latency:.2f}ms should be under 25ms threshold"
    assert latency > 0.0, "Latency should be positive"
