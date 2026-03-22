from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import networkx as nx


@dataclass
class Drone:
    """Represents a single enemy drone in the swarm."""

    id: int
    type: str  # 'kamikaze' | 'isr' | 'jammer'
    position: np.ndarray  # 3D [x, y, z]
    velocity: np.ndarray  # 3D [vx, vy, vz]
    health: float
    is_alive: bool
    rf_emission_power: float
    formation_id: int

    def __post_init__(self) -> None:
        self.position = np.array(self.position, dtype=np.float32)
        self.velocity = np.array(self.velocity, dtype=np.float32)


# Speed and RF emission power per drone type
_TYPE_SPEED: Dict[str, float] = {"kamikaze": 40.0, "isr": 15.0, "jammer": 8.0}
_TYPE_HEALTH: Dict[str, float] = {"kamikaze": 1.0, "isr": 1.5, "jammer": 2.0}
_TYPE_RF: Dict[str, float] = {"kamikaze": 0.1, "isr": 0.5, "jammer": 2.0}

# Formation tactics
_TACTICS = ("line", "pincer", "dispersal", "feint_and_strike", "spiral")


class DroneSwarm:
    """
    Manages the heterogeneous enemy drone swarm including spawning,
    formation tactics, attrition, and graph representation.
    """

    def __init__(
        self,
        n_drones: int = 10,
        type_distribution: Optional[Dict[str, int]] = None,
        formation: str = "line",
    ) -> None:
        if type_distribution is None:
            type_distribution = {"kamikaze": 5, "isr": 3, "jammer": 2}
        assert (
            sum(type_distribution.values()) == n_drones
        ), "type_distribution counts must sum to n_drones"
        self.n_drones = n_drones
        self.type_distribution = type_distribution
        self.formation = formation
        self.drones: List[Drone] = []
        self._drone_id_counter = 0

    def spawn_swarm(self, center: np.ndarray, spread: float = 50.0) -> List[Drone]:
        """Spawn all drones around a center point with given spread."""
        self.drones = []
        self._drone_id_counter = 0
        formation_id = 0
        rng = np.random.default_rng(42)

        for drone_type, count in self.type_distribution.items():
            speed = _TYPE_SPEED[drone_type]
            health = _TYPE_HEALTH[drone_type]
            rf_power = _TYPE_RF[drone_type]
            for _ in range(count):
                offset = rng.uniform(-spread, spread, size=3)
                offset[2] = abs(offset[2])  # altitude positive
                pos = np.array(center, dtype=np.float32) + offset.astype(np.float32)
                # Initial velocity directed roughly toward origin (0,0,0)
                direction = -pos / (np.linalg.norm(pos) + 1e-8)
                vel = direction * speed + rng.standard_normal(3).astype(np.float32) * 2.0
                drone = Drone(
                    id=self._drone_id_counter,
                    type=drone_type,
                    position=pos,
                    velocity=vel.astype(np.float32),
                    health=health,
                    is_alive=True,
                    rf_emission_power=rf_power,
                    formation_id=formation_id,
                )
                self.drones.append(drone)
                self._drone_id_counter += 1
            formation_id += 1

        return self.drones

    def update_formation(self, tactic: str) -> None:
        """Update swarm formation according to the specified tactic."""
        assert tactic in _TACTICS, f"tactic must be one of {_TACTICS}"
        self.formation = tactic
        alive = [d for d in self.drones if d.is_alive]
        n = len(alive)
        if n == 0:
            return

        center = np.mean([d.position for d in alive], axis=0)

        if tactic == "line":
            for i, drone in enumerate(alive):
                offset = np.array([(i - n / 2) * 10.0, 0.0, 0.0], dtype=np.float32)
                drone.position = center + offset

        elif tactic == "pincer":
            half = n // 2
            for i, drone in enumerate(alive[:half]):
                drone.position = center + np.array(
                    [-20.0, (i - half / 2) * 15.0, 0.0], dtype=np.float32
                )
            for i, drone in enumerate(alive[half:]):
                drone.position = center + np.array(
                    [20.0, (i - (n - half) / 2) * 15.0, 0.0], dtype=np.float32
                )

        elif tactic == "dispersal":
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            radius = 40.0
            for drone, angle in zip(alive, angles):
                drone.position = center + np.array(
                    [radius * np.cos(angle), radius * np.sin(angle), 0.0], dtype=np.float32
                )

        elif tactic == "feint_and_strike":
            feint_group = alive[: n // 3]
            strike_group = alive[n // 3 :]
            for i, drone in enumerate(feint_group):
                drone.position = center + np.array(
                    [0.0, (i - len(feint_group) / 2) * 20.0, 20.0], dtype=np.float32
                )
            for i, drone in enumerate(strike_group):
                drone.position = center + np.array(
                    [(i - len(strike_group) / 2) * 15.0, 0.0, 5.0], dtype=np.float32
                )

        elif tactic == "spiral":
            for i, drone in enumerate(alive):
                angle = i * 0.6
                r = 5.0 + i * 4.0
                drone.position = center + np.array(
                    [r * np.cos(angle), r * np.sin(angle), i * 1.5], dtype=np.float32
                )

    def apply_attrition(self, neutralized_ids: List[int]) -> None:
        """Mark drones as neutralized and reduce their health to 0."""
        id_set = set(neutralized_ids)
        for drone in self.drones:
            if drone.id in id_set:
                drone.is_alive = False
                drone.health = 0.0

    def reform_after_loss(self, neutralized_ids: List[int]) -> None:
        """
        Redistribute roles if critical drones (jammer) are lost.
        The highest-health remaining drone assumes the jammer role.
        """
        self.apply_attrition(neutralized_ids)
        alive = [d for d in self.drones if d.is_alive]
        jammer_alive = any(d.type == "jammer" for d in alive)
        if not jammer_alive and alive:
            # Promote the highest-health drone to jammer role
            best = max(alive, key=lambda d: d.health)
            best.type = "jammer"
            best.rf_emission_power = _TYPE_RF["jammer"]

    def get_swarm_graph(self) -> nx.DiGraph:
        """Build a directed NetworkX graph of the live swarm."""
        G = nx.DiGraph()
        alive = [d for d in self.drones if d.is_alive]
        for drone in alive:
            G.add_node(
                drone.id,
                pos=drone.position.tolist(),
                type=drone.type,
                health=drone.health,
                vel=drone.velocity.tolist(),
            )
        # Add edges between drones within communication range (100m)
        for i, d1 in enumerate(alive):
            for d2 in alive[i + 1 :]:
                dist = float(np.linalg.norm(d1.position - d2.position))
                if dist < 100.0:
                    G.add_edge(d1.id, d2.id, weight=1.0 / (dist + 1.0))
                    G.add_edge(d2.id, d1.id, weight=1.0 / (dist + 1.0))
        return G

    def compute_threat_vector(self) -> np.ndarray:
        """Compute aggregated threat vector from all alive drones."""
        alive = [d for d in self.drones if d.is_alive]
        if not alive:
            return np.zeros(6, dtype=np.float32)
        positions = np.stack([d.position for d in alive])
        velocities = np.stack([d.velocity for d in alive])
        centroid = positions.mean(axis=0)
        mean_vel = velocities.mean(axis=0)
        return np.concatenate([centroid, mean_vel]).astype(np.float32)

    def get_state_vector(self) -> np.ndarray:
        """Return full swarm state as a flat numpy array."""
        states = []
        for drone in self.drones:
            alive_flag = 1.0 if drone.is_alive else 0.0
            type_enc = {"kamikaze": 0, "isr": 1, "jammer": 2}[drone.type]
            states.extend(
                [
                    *drone.position,
                    *drone.velocity,
                    drone.health,
                    alive_flag,
                    drone.rf_emission_power,
                    float(type_enc),
                ]
            )
        return np.array(states, dtype=np.float32)

    @property
    def alive_drones(self) -> List[Drone]:
        return [d for d in self.drones if d.is_alive]

    @property
    def n_alive(self) -> int:
        return sum(1 for d in self.drones if d.is_alive)


if __name__ == "__main__":
    swarm = DroneSwarm(n_drones=10)
    drones = swarm.spawn_swarm(center=np.array([400.0, 400.0, 50.0]))
    print(f"Spawned {len(drones)} drones")
    swarm.update_formation("pincer")
    print(f"Formation: {swarm.formation}")
    G = swarm.get_swarm_graph()
    print(f"Swarm graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("drone_swarm.py OK")
