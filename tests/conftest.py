"""
Shared pytest fixtures for SWARM-SHIELD tests.
"""

import os
import sys

import pytest

# Ensure repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.commander_agent import CommanderAgent  # noqa: E402
from adversarial.attacker_agent import QMIXAttacker  # noqa: E402
from envs.cuas_env import CUASEnv  # noqa: E402
from models.gat_network import SwarmGAT  # noqa: E402
from models.snn_network import SNNNetwork  # noqa: E402


@pytest.fixture
def env():
    """CUASEnv instance with default config."""
    config = {
        "grid_size": 500,
        "n_enemy_drones": 10,
        "drone_types": {
            "kamikaze": {"count": 5, "speed": 40, "health": 1.0},
            "isr": {"count": 3, "speed": 15, "health": 1.5},
            "jammer": {"count": 2, "speed": 8, "health": 2.0},
        },
        "n_interceptors": 4,
        "max_timesteps": 300,
        "n_jammers": 2,
        "jammer_radius": 80,
        "jammer_power_dbm": 45,
        "gps_denial_zones": 2,
        "communication_radius": 150,
        "radar_range": 300,
        "eo_ir_range": 200,
    }
    e = CUASEnv(config=config)
    yield e
    e.close()


@pytest.fixture
def snn():
    """SNNNetwork instance."""
    return SNNNetwork(input_dim=128, output_dim=64)


@pytest.fixture
def gat():
    """SwarmGAT instance."""
    return SwarmGAT(node_feat_dim=64)


@pytest.fixture
def commander():
    """CommanderAgent instance with obs_dim matching typical usage."""
    return CommanderAgent(
        obs_dim=128,
        action_dim=11,
        n_targets=10,
        n_interceptors=4,
    )


@pytest.fixture
def attacker():
    """QMIXAttacker instance."""
    return QMIXAttacker(
        n_agents=10,
        obs_dim=64,
        action_dim=6,
        state_dim=128,
    )
