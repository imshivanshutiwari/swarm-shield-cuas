"""
Tests for adversarial components: QMIX, Stackelberg, Curriculum.
"""

import torch

from adversarial.curriculum import CurriculumScheduler
from agents.stackelberg import StackelbergGame


def test_qmix_monotonicity_constraint(attacker):
    """
    QMIX monotonicity: mixing network uses abs weights.
    Total Q should increase when any agent Q increases.
    """
    B = 4
    state = torch.randn(B, 128)

    # Use a baseline agent_qs vector
    agent_qs_base = torch.zeros(B, attacker.n_agents)

    # Increase one agent's Q-value
    agent_qs_higher = agent_qs_base.clone()
    agent_qs_higher[:, 0] += 5.0

    with torch.no_grad():
        total_q_base = attacker.mixing_network(agent_qs_base, state)
        total_q_higher = attacker.mixing_network(agent_qs_higher, state)

    # With monotonicity (abs weights), higher agent Q => higher total Q
    assert (
        total_q_higher >= total_q_base - 1e-4
    ).all(), "QMIX monotonicity violated: total Q should increase with higher agent Q"


def test_stackelberg_update_schedule():
    """Attacker should update exactly 3 times after 9 defender steps."""
    game = StackelbergGame(n_attacker_agents=10, n_defender_agents=4)
    attacker_updates = 0
    for ep in range(9):
        if game.update_schedule(ep):
            attacker_updates += 1
    assert attacker_updates == 3, f"Expected 3 attacker updates in 9 steps, got {attacker_updates}"


def test_curriculum_phase_transitions():
    """Curriculum phases should return correct n_enemy_drones."""
    scheduler = CurriculumScheduler()

    config_p1 = scheduler.get_config(0)
    config_p2 = scheduler.get_config(500001)
    config_p3 = scheduler.get_config(1500001)

    assert (
        config_p1["n_enemy_drones"] == 3
    ), f"Phase 1 should have 3 drones, got {config_p1['n_enemy_drones']}"
    assert (
        config_p2["n_enemy_drones"] == 7
    ), f"Phase 2 should have 7 drones, got {config_p2['n_enemy_drones']}"
    assert (
        config_p3["n_enemy_drones"] == 10
    ), f"Phase 3 should have 10 drones, got {config_p3['n_enemy_drones']}"
