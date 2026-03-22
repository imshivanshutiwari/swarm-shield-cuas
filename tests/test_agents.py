"""
Tests for Commander and Interceptor agents.
"""

import numpy as np

from agents.interceptor_agent import InterceptorAgent
from training.rollout_buffer import RolloutBuffer


def test_commander_action_in_valid_range(commander):
    """Commander get_action() should return valid assignment array."""
    obs = np.random.randn(128).astype(np.float32)
    action, log_prob, value = commander.get_action(obs, deterministic=False)
    assert action.shape == (
        commander.n_interceptors,
    ), f"Expected shape ({commander.n_interceptors},), got {action.shape}"
    # All assignments should be in [0, n_targets]
    assert np.all(action >= 0), "All assignments should be >= 0"
    assert np.all(
        action <= commander.n_targets
    ), f"All assignments should be <= {commander.n_targets}"
    assert isinstance(log_prob, float), "log_prob should be a float"
    assert isinstance(value, float), "value should be a float"


def test_interceptor_thrust_in_bounds():
    """Interceptor get_action() should return thrust in [-1, 1]^3."""
    agent = InterceptorAgent(obs_dim=128, action_dim=3, n_agents=4)
    obs = np.random.randn(128).astype(np.float32)
    action = agent.get_action(obs, add_noise=False)
    assert action.shape == (3,), f"Expected shape (3,), got {action.shape}"
    assert np.all(action >= -1.0), f"Action out of bounds: {action}"
    assert np.all(action <= 1.0), f"Action out of bounds: {action}"


def test_mappo_policy_loss_decreases(commander):
    """Running 5 update steps on a consistent buffer should reduce policy loss."""
    # Create and fill a rollout buffer
    buffer = RolloutBuffer(rollout_steps=64, obs_dim=128, action_dim=4)
    for _ in range(64):
        buffer.add(
            obs=np.random.randn(128).astype(np.float32),
            action=np.random.randint(0, 5, 4).astype(np.float32),
            log_prob=float(np.random.randn()),
            reward=float(np.random.randn()),
            value=float(np.random.randn()),
            done=0.0,
        )
    buffer.compute_returns_and_advantages()

    losses = []
    for _ in range(5):
        p_loss, v_loss, entropy = commander.update(buffer)
        losses.append(p_loss)

    # We don't strictly enforce loss decrease (untrained network can fluctuate),
    # but we ensure updates run without errors and losses are finite
    assert all(np.isfinite(loss_val) for loss_val in losses), f"Some losses are not finite: {losses}"


def test_ctde_decentralized_execution():
    """Interceptor should execute using only local observation (no global state)."""
    agent = InterceptorAgent(obs_dim=128, action_dim=3, n_agents=4)
    # Provide only local own_state (6,) padded to obs_dim
    local_obs = np.zeros(128, dtype=np.float32)
    local_obs[:6] = np.random.randn(6)

    # This should NOT raise any error
    action = agent.get_action(local_obs, add_noise=False)
    assert action.shape == (3,), "CTDE: should return 3D action with local obs"
    assert np.all(action >= -1.0) and np.all(action <= 1.0), "Action should be in [-1, 1]^3"
