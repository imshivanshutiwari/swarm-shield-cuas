"""
Tests for CUASEnv environment.
"""

import numpy as np

from envs.jammer_module import EWJammer


def test_reset_returns_correct_obs_shape(env):
    """reset() should return observations with all expected Dict keys and shapes."""
    obs, info = env.reset(seed=42)
    assert "commander" in obs
    for i in range(env.n_interceptors):
        key = f"interceptor_{i}"
        assert key in obs, f"Missing key: {key}"
        int_obs = obs[key]
        assert "radar_returns" in int_obs
        assert "gat_neighbor_obs" in int_obs
        assert "rf_spectrogram" in int_obs
        assert "own_state" in int_obs
        assert "commander_task" in int_obs
        assert int_obs["radar_returns"].shape == (env.n_targets, 4)
        assert int_obs["gat_neighbor_obs"].shape == (env.n_interceptors - 1, 64)
        assert int_obs["rf_spectrogram"].shape == (64, 64)
        assert int_obs["own_state"].shape == (6,)


def test_heterogeneous_drones_initialized(env):
    """Environment should spawn exactly 5 kamikaze, 3 isr, 2 jammer drones."""
    env.reset(seed=0)
    assert env.swarm is not None
    drones = env.swarm.drones
    kamikaze_count = sum(1 for d in drones if d.type == "kamikaze")
    isr_count = sum(1 for d in drones if d.type == "isr")
    jammer_count = sum(1 for d in drones if d.type == "jammer")
    assert kamikaze_count == 5, f"Expected 5 kamikaze, got {kamikaze_count}"
    assert isr_count == 3, f"Expected 3 isr, got {isr_count}"
    assert jammer_count == 2, f"Expected 2 jammer, got {jammer_count}"


def test_jamming_corrupts_radar_snr(env):
    """Placing interceptor in jammer zone should reduce radar SNR."""
    env.reset(seed=1)
    # Place a jammer right on interceptor 0's position
    jammer_pos = env.interceptor_positions[0].copy()
    test_jammer = EWJammer(
        position=jammer_pos,
        radius=200.0,
        power_dbm=60.0,
        jam_type="broadband",
    )
    # Compute baseline SNR with no jamming
    baseline_obs = env._get_radar_returns(0)
    baseline_snr = baseline_obs[:, 3].mean()

    # Corrupt radar returns using the jammer
    effect = test_jammer.compute_jamming_effect(jammer_pos)
    assert effect > 0, "Jammer should have effect at zero distance"

    corrupted = np.array(
        [test_jammer.corrupt_radar_return(baseline_obs[t], effect) for t in range(env.n_targets)]
    )
    jammed_snr = corrupted[:, 3].mean()
    assert (
        jammed_snr <= baseline_snr
    ), f"Jammed SNR {jammed_snr:.3f} should be <= baseline {baseline_snr:.3f}"


def test_gps_denial_adds_drift(env):
    """Running 50 steps in GPS denial zone should accumulate position error."""
    # Place GPS denial zone on interceptor 0
    env.reset(seed=2)
    env.gps_denial_centers = [env.interceptor_positions[0].copy()]

    initial_drift = env.gps_drift_acc[0].copy()
    n_steps = 50

    actions = {f"interceptor_{i}": np.zeros(3, dtype=np.float32) for i in range(env.n_interceptors)}
    actions["commander"] = np.zeros(env.n_interceptors, dtype=int)

    for _ in range(n_steps):
        env.step(actions)

    final_drift = env.gps_drift_acc[0]
    drift_magnitude = np.linalg.norm(final_drift - initial_drift)
    assert drift_magnitude > 0, "GPS drift should accumulate in denial zone"


def test_commander_action_assigns_interceptors(env):
    """Commander action should be accepted and processed by the environment."""
    obs, _ = env.reset(seed=3)
    # Assign each interceptor to a different target
    cmd_action = np.array([0, 1, 2, 3], dtype=int)
    actions = {
        "commander": cmd_action,
        **{
            f"interceptor_{i}": np.random.uniform(-1, 1, 3).astype(np.float32)
            for i in range(env.n_interceptors)
        },
    }
    obs2, rewards, terminated, truncated, info = env.step(actions)
    assert "commander" in rewards, "Commander reward should be in rewards dict"
    for i in range(env.n_interceptors):
        assert f"interceptor_{i}" in rewards


def test_episode_terminates_when_swarm_cleared(env):
    """Manually neutralizing all drones should result in terminated=True."""
    env.reset(seed=4)
    # Neutralize all drones manually
    for drone in env.swarm.drones:
        drone.is_alive = False
        drone.health = 0.0
    env.neutralized_count = env.n_enemy_drones

    # One step to trigger termination check
    actions = {
        "commander": np.zeros(env.n_interceptors, dtype=int),
        **{f"interceptor_{i}": np.zeros(3, dtype=np.float32) for i in range(env.n_interceptors)},
    }
    _, _, terminated, _, _ = env.step(actions)
    assert terminated, "Episode should terminate when all enemy drones are neutralized"
