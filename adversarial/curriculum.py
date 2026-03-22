from typing import Any, Dict, List

import numpy as np


class CurriculumScheduler:
    """
    Three-phase curriculum scheduler for progressive difficulty.
    Phase 1 (0 → 500K steps):    Easy - few drones, no EW
    Phase 2 (500K → 1.5M steps): Medium - more drones, EW enabled
    Phase 3 (1.5M → 3M steps):   Hard - full swarm, adversarial attacker
    """

    PHASE1_END = 500_000
    PHASE2_END = 1_500_000
    PHASE3_END = 3_000_000

    def __init__(self) -> None:
        self._phase_metrics: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: []}

    def get_phase(self, step: int) -> int:
        """Return current curriculum phase (1, 2, or 3)."""
        if step < self.PHASE1_END:
            return 1
        elif step < self.PHASE2_END:
            return 2
        else:
            return 3

    def get_config(self, step: int) -> Dict[str, Any]:
        """Return environment configuration for the given training step."""
        phase = self.get_phase(step)

        if phase == 1:
            return {
                "grid_size": 500,
                "n_enemy_drones": 3,
                "drone_types": {
                    "kamikaze": {"count": 2, "speed": 40, "health": 1.0},
                    "isr": {"count": 1, "speed": 15, "health": 1.5},
                    "jammer": {"count": 0, "speed": 8, "health": 2.0},
                },
                "n_interceptors": 4,
                "max_timesteps": 300,
                "n_jammers": 0,
                "jammer_radius": 80,
                "jammer_power_dbm": 45,
                "gps_denial_zones": 0,
                "communication_radius": 150,
                "radar_range": 300,
                "eo_ir_range": 200,
                "jamming": False,
                "gps_denial": False,
                "swarm_tactics": ["line"],
                "adversarial_attacker": False,
                "curriculum_phase": 1,
            }

        elif phase == 2:
            return {
                "grid_size": 500,
                "n_enemy_drones": 7,
                "drone_types": {
                    "kamikaze": {"count": 3, "speed": 40, "health": 1.0},
                    "isr": {"count": 2, "speed": 15, "health": 1.5},
                    "jammer": {"count": 2, "speed": 8, "health": 2.0},
                },
                "n_interceptors": 4,
                "max_timesteps": 300,
                "n_jammers": 2,
                "jammer_radius": 80,
                "jammer_power_dbm": 45,
                "gps_denial_zones": 1,
                "communication_radius": 150,
                "radar_range": 300,
                "eo_ir_range": 200,
                "jamming": True,
                "gps_denial": True,
                "swarm_tactics": ["line", "pincer", "dispersal"],
                "adversarial_attacker": False,
                "curriculum_phase": 2,
            }

        else:  # phase 3
            return {
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
                "jamming": True,
                "gps_denial": True,
                "swarm_tactics": ["line", "pincer", "dispersal", "feint_and_strike", "spiral"],
                "adversarial_attacker": True,
                "curriculum_phase": 3,
            }

    def is_phase_transition(self, step: int) -> bool:
        """Return True if this step marks a phase boundary."""
        return step == self.PHASE1_END or step == self.PHASE2_END

    def log_phase_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        """Record metrics for current phase."""
        phase = self.get_phase(step)
        self._phase_metrics[phase].append({"step": step, **metrics})

    def get_phase_summary(self, phase: int) -> Dict[str, Any]:
        """Return summary statistics for a given phase."""
        records = self._phase_metrics.get(phase, [])
        if not records:
            return {}
        rewards = [r.get("mean_reward", 0.0) for r in records]
        return {
            "phase": phase,
            "n_episodes": len(records),
            "mean_reward": float(np.mean(rewards)),
            "max_reward": float(np.max(rewards)),
        }


if __name__ == "__main__":
    scheduler = CurriculumScheduler()
    print("Phase 1 config:", scheduler.get_config(0)["n_enemy_drones"])
    print("Phase 2 config:", scheduler.get_config(500001)["n_enemy_drones"])
    print("Phase 3 config:", scheduler.get_config(1500001)["n_enemy_drones"])
    assert scheduler.get_config(0)["n_enemy_drones"] == 3
    assert scheduler.get_config(500001)["n_enemy_drones"] == 7
    assert scheduler.get_config(1500001)["n_enemy_drones"] == 10
    print("curriculum.py OK")
