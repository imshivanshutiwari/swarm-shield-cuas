"""
Evaluation metrics for SWARM-SHIELD.
All metrics implemented with correct mathematical formulas.
"""

from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment


def swarm_neutralization_rate(neutralized: int, total: int) -> float:
    """
    Compute neutralization rate as a percentage.
    neutralization_rate = (neutralized / total) * 100
    """
    if total <= 0:
        return 0.0
    return float(neutralized / total * 100.0)


def mean_time_to_neutralize(engagement_log: List[Dict]) -> float:
    """
    Compute mean time (in seconds) to neutralize each target.
    engagement_log: list of dicts with 'step' and 'target' keys.
    Returns mean time assuming 0.1s per step (100ms timestep).
    """
    if not engagement_log:
        return 0.0
    times = [e.get("step", 0) * 0.1 for e in engagement_log]
    return float(np.mean(times))


def friendly_fire_rate(ff_events: int, total_engagements: int) -> float:
    """
    Compute friendly fire rate as a percentage.
    friendly_fire_rate = (ff_events / total_engagements) * 100
    """
    if total_engagements <= 0:
        return 0.0
    return float(ff_events / total_engagements * 100.0)


def engagement_latency_ms(action_timestamps: List[float]) -> float:
    """
    Compute mean engagement latency in milliseconds.
    Latency = time between target detection and first engagement action.
    action_timestamps: list of timestamps in seconds.
    Returns mean delta in milliseconds.
    """
    if len(action_timestamps) < 2:
        return 0.0
    deltas = np.diff(action_timestamps)
    return float(np.mean(deltas) * 1000.0)


def snn_spike_efficiency(spike_counts: Dict[str, float], total_ops: int) -> float:
    """
    Compute SNN spike efficiency.
    efficiency = 1 - (total_spikes / total_ops)
    Returns value in [0, 1] where 1 = maximum efficiency.
    """
    if total_ops <= 0:
        return 1.0
    total_spikes = sum(spike_counts.values()) if spike_counts else 0
    return float(1.0 - min(total_spikes / total_ops, 1.0))


def ospa_distance(
    estimated_positions: np.ndarray,
    true_positions: np.ndarray,
    c: float = 20.0,
    p: int = 1,
) -> float:
    """
    Compute OSPA (Optimal Sub-Pattern Assignment) distance.

    OSPA(X, Y) = (1/max(m,n) * (min_perm sum d_c(x_i, y_j)^p + c^p * |m-n|)) ^ (1/p)
    where d_c(x,y) = min(c, ||x-y||)

    Args:
        estimated_positions: (m, D) array of estimated positions
        true_positions: (n, D) array of true positions
        c: cut-off parameter (maximum distance contribution per point)
        p: order parameter (1 = OSPA, 2 = OSPA^2)
    Returns:
        OSPA distance (scalar)
    """
    estimated_positions = np.array(estimated_positions)
    true_positions = np.array(true_positions)

    m = len(estimated_positions)
    n = len(true_positions)

    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return float(c)

    # Build cost matrix: d_c(x_i, y_j)^p
    cost_matrix = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            dist = np.linalg.norm(estimated_positions[i] - true_positions[j])
            cost_matrix[i, j] = min(c, dist) ** p

    # Solve optimal assignment
    max_mn = max(m, n)
    if m <= n:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignment_cost = cost_matrix[row_ind, col_ind].sum()
    else:
        row_ind, col_ind = linear_sum_assignment(cost_matrix.T)
        assignment_cost = cost_matrix.T[row_ind, col_ind].sum()

    # Cardinality penalty
    cardinality_penalty = (c**p) * abs(m - n)

    ospa = ((assignment_cost + cardinality_penalty) / max_mn) ** (1.0 / p)
    return float(ospa)


def gospa_metric(
    estimated: np.ndarray,
    true: np.ndarray,
    c: float = 20.0,
    p: int = 2,
    alpha: int = 2,
) -> float:
    """
    Compute GOSPA (Generalized OSPA) metric.
    GOSPA decomposes into: localization + missed detections + false alarms.

    Args:
        estimated: (m, D) estimated positions
        true: (n, D) true positions
        c: cut-off distance
        p: order parameter
        alpha: parameter in {1, 2} (2 for decomposed form)
    Returns:
        GOSPA distance (scalar)
    """
    estimated = np.array(estimated)
    true = np.array(true)

    m = len(estimated)
    n = len(true)

    if m == 0 and n == 0:
        return 0.0
    if m == 0:
        return float((c**p / alpha) * n) ** (1.0 / p)
    if n == 0:
        return float((c**p / alpha) * m) ** (1.0 / p)

    # Build cost matrix
    cost_matrix = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            dist = np.linalg.norm(estimated[i] - true[j])
            cost_matrix[i, j] = min(c, dist) ** p

    # Solve assignment for min(m, n) pairs
    if m <= n:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    else:
        row_ind, col_ind = linear_sum_assignment(cost_matrix.T)
        row_ind, col_ind = col_ind, row_ind

    n_assigned = len(row_ind)
    assignment_cost = cost_matrix[row_ind, col_ind].sum()

    # Missed detections and false alarms
    n_missed = n - n_assigned
    n_false = m - n_assigned
    penalty = (c**p / alpha) * (n_missed + n_false)

    gospa = (assignment_cost + penalty) ** (1.0 / p)
    return float(gospa)


def jamming_resilience_score(baseline_snr: float, jammed_snr: float) -> float:
    """
    Compute jamming resilience score.
    Score = jammed_snr / baseline_snr in [0, 1].
    1.0 = fully resilient (no SNR loss), 0.0 = completely jammed.
    """
    if baseline_snr <= 0:
        return 1.0
    return float(np.clip(jammed_snr / baseline_snr, 0.0, 1.0))


def nash_convergence_episodes(reward_history: List[float], window: int = 100) -> int:
    """
    Estimate the number of episodes until Nash equilibrium convergence.
    Uses a rolling window variance threshold.
    Returns the first episode index where variance drops below threshold.
    """
    if len(reward_history) < window:
        return len(reward_history)

    rewards = np.array(reward_history)
    threshold = np.std(rewards) * 0.1  # 10% of global std

    for i in range(window, len(rewards)):
        window_var = np.std(rewards[i - window : i])
        if window_var < threshold:
            return i

    return len(reward_history)


if __name__ == "__main__":
    # Verify all metrics
    print(f"Neutralization rate: {swarm_neutralization_rate(8, 10)}")
    assert swarm_neutralization_rate(8, 10) == 80.0

    est = np.array([[0.0, 0.0], [1.0, 0.0]])
    true_pos = np.array([[0.0, 0.0], [1.0, 0.0]])
    ospa = ospa_distance(est, true_pos)
    print(f"OSPA (perfect): {ospa}")

    est2 = np.array([[100.0, 100.0]])
    true_pos2 = np.array([[0.0, 0.0]])
    ospa2 = ospa_distance(est2, true_pos2, c=20.0)
    print(f"OSPA (far): {ospa2}")

    timestamps = [0.001 * i for i in range(5)]
    lat = engagement_latency_ms(timestamps)
    print(f"Engagement latency: {lat:.2f} ms")
    assert lat < 25.0

    print("metrics.py OK")
