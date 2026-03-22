from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from envs.drone_swarm import DroneSwarm
from envs.jammer_module import EWJammer
from envs.terrain_map import TerrainMap
from models.gat_network import SwarmGAT


class CUASEnv(gym.Env):
    """
    Counter-UAS environment with heterogeneous drone swarm.
    Implements a full Gymnasium environment for multi-agent MARL training.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Reward constants
    CMD_NEUTRALIZE = 15.0
    CMD_FRIENDLY_FIRE = -30.0
    CMD_MISSED = -5.0
    CMD_TIME = -0.2
    CMD_CLEAR_BONUS = 50.0
    INT_HIT = 10.0
    INT_MISS = -2.0
    INT_OUT_OF_ZONE = -1.0
    INT_TIME = -0.1
    INT_COOP_BONUS = 5.0

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        if config is None:
            config = self._default_config()
        self.config = config

        self.grid_size: int = config.get("grid_size", 500)
        self.n_enemy_drones: int = config.get("n_enemy_drones", 10)
        self.n_interceptors: int = config.get("n_interceptors", 4)
        self.max_timesteps: int = config.get("max_timesteps", 300)
        self.n_jammers_hw: int = config.get("n_jammers", 2)
        self.jammer_radius: float = float(config.get("jammer_radius", 80))
        self.jammer_power_dbm: float = float(config.get("jammer_power_dbm", 45))
        self.gps_denial_zones_count: int = config.get("gps_denial_zones", 2)
        self.comm_radius: float = float(config.get("communication_radius", 150))
        self.radar_range: float = float(config.get("radar_range", 300))
        self.eo_ir_range: float = float(config.get("eo_ir_range", 200))

        drone_types = config.get(
            "drone_types",
            {
                "kamikaze": {"count": 5, "speed": 40, "health": 1.0},
                "isr": {"count": 3, "speed": 15, "health": 1.5},
                "jammer": {"count": 2, "speed": 8, "health": 2.0},
            },
        )
        self.type_distribution = {k: v["count"] for k, v in drone_types.items()}
        assert sum(self.type_distribution.values()) == self.n_enemy_drones

        # Observation and action spaces
        self.n_targets = self.n_enemy_drones
        gat_dim = 64

        # Per-interceptor observation space
        single_obs_space = spaces.Dict(
            {
                "radar_returns": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.n_targets, 4), dtype=np.float32
                ),
                "gat_neighbor_obs": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.n_interceptors - 1, gat_dim),
                    dtype=np.float32,
                ),
                "rf_spectrogram": spaces.Box(low=0.0, high=1.0, shape=(64, 64), dtype=np.float32),
                "own_state": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
                "commander_task": spaces.Discrete(self.n_targets + 1),
            }
        )

        self.observation_space = spaces.Dict(
            {f"interceptor_{i}": single_obs_space for i in range(self.n_interceptors)}
        )
        self.observation_space["commander"] = spaces.Dict(
            {
                "swarm_state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.n_enemy_drones * 12,),
                    dtype=np.float32,
                ),
                "interceptor_states": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.n_interceptors * 6,),
                    dtype=np.float32,
                ),
            }
        )

        # Commander action: assign each interceptor to a target
        self.commander_action_space = spaces.MultiDiscrete(
            [self.n_targets + 1] * self.n_interceptors
        )
        # Interceptor action: 3D thrust
        self.interceptor_action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # Combined action space
        self.action_space = self.interceptor_action_space  # primary

        # Internal state
        self.swarm: Optional[DroneSwarm] = None
        self.interceptor_positions: Optional[np.ndarray] = None
        self.interceptor_velocities: Optional[np.ndarray] = None
        self.interceptor_health: Optional[np.ndarray] = None
        self.hw_jammers: List[EWJammer] = []
        self.gps_denial_centers: List[np.ndarray] = []
        self.terrain: Optional[TerrainMap] = None
        self.gat_model: Optional[SwarmGAT] = None
        self.current_step: int = 0
        self.commander_assignments: np.ndarray = np.zeros(self.n_interceptors, dtype=int)
        self.gps_drift_acc: np.ndarray = np.zeros((self.n_interceptors, 3), dtype=np.float32)
        self.friendly_fire_count: int = 0
        self.neutralized_count: int = 0
        self.engagement_log: List[Dict] = []

        # Render state
        self._render_fig = None
        self._render_ax = None

    @staticmethod
    def _default_config() -> Dict[str, Any]:
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
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment and return initial observations."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.commander_assignments = np.zeros(self.n_interceptors, dtype=int)
        self.gps_drift_acc = np.zeros((self.n_interceptors, 3), dtype=np.float32)
        self.friendly_fire_count = 0
        self.neutralized_count = 0
        self.engagement_log = []

        # Initialize terrain
        s = seed if seed is not None else 42
        self.terrain = TerrainMap(grid_size=self.grid_size, seed=s)

        # Spawn enemy swarm
        self.swarm = DroneSwarm(
            n_drones=self.n_enemy_drones,
            type_distribution=self.type_distribution,
        )
        spawn_center = np.array(
            [
                np.random.uniform(350, 480),
                np.random.uniform(350, 480),
                np.random.uniform(30, 80),
            ],
            dtype=np.float32,
        )
        self.swarm.spawn_swarm(center=spawn_center, spread=40.0)

        # Initialize interceptors
        self.interceptor_positions = np.zeros((self.n_interceptors, 3), dtype=np.float32)
        self.interceptor_velocities = np.zeros((self.n_interceptors, 3), dtype=np.float32)
        self.interceptor_health = np.ones(self.n_interceptors, dtype=np.float32)
        for i in range(self.n_interceptors):
            self.interceptor_positions[i] = [
                np.random.uniform(20, 100),
                np.random.uniform(20, 100) + i * 20,
                np.random.uniform(20, 60),
            ]

        # Place EW jammers
        self.hw_jammers = []
        for _ in range(self.n_jammers_hw):
            jammer_pos = np.array(
                [
                    np.random.uniform(100, 400),
                    np.random.uniform(100, 400),
                    0.0,
                ],
                dtype=np.float32,
            )
            self.hw_jammers.append(
                EWJammer(
                    position=jammer_pos,
                    radius=self.jammer_radius,
                    power_dbm=self.jammer_power_dbm,
                    jam_type="broadband",
                )
            )

        # GPS denial zones
        self.gps_denial_centers = [
            np.array(
                [
                    np.random.uniform(50, 450),
                    np.random.uniform(50, 450),
                    0.0,
                ],
                dtype=np.float32,
            )
            for _ in range(self.gps_denial_zones_count)
        ]

        # Initialize GAT
        self.gat_model = SwarmGAT(node_feat_dim=64)

        obs = self._get_observations()
        info = {
            "n_alive_drones": self.swarm.n_alive,
            "step": self.current_step,
        }
        return obs, info

    def step(self, actions_dict: Dict[str, Any]) -> Tuple[Dict, Dict, bool, bool, Dict]:
        """Execute actions and return next state."""
        self.current_step += 1

        # Process commander assignment
        if "commander" in actions_dict:
            cmd_action = actions_dict["commander"]
            if hasattr(cmd_action, "tolist"):
                self.commander_assignments = np.array(cmd_action)
            else:
                self.commander_assignments = np.array(cmd_action)

        # Process interceptor actions (thrust)
        interceptor_rewards = np.zeros(self.n_interceptors, dtype=np.float32)
        commander_reward = float(self.CMD_TIME)

        alive_drones = self.swarm.alive_drones
        alive_ids = [d.id for d in alive_drones]

        for i in range(self.n_interceptors):
            key = f"interceptor_{i}"
            if key in actions_dict:
                thrust = np.array(actions_dict[key], dtype=np.float32)
                thrust = np.clip(thrust, -1.0, 1.0)
            else:
                thrust = np.zeros(3, dtype=np.float32)

            # Update interceptor velocity and position
            speed = 20.0  # m/step
            self.interceptor_velocities[i] = thrust * speed
            self.interceptor_positions[i] += self.interceptor_velocities[i]
            self.interceptor_positions[i] = np.clip(
                self.interceptor_positions[i], 0.0, float(self.grid_size)
            )

            # Apply GPS denial drift
            in_denial = self._in_gps_denial_zone(self.interceptor_positions[i])
            if in_denial:
                drift = np.random.normal(0, 0.5, size=3).astype(np.float32)
                self.gps_drift_acc[i] += drift

            # Check engagement
            target_idx = (
                int(self.commander_assignments[i]) if i < len(self.commander_assignments) else 0
            )
            target_drone = None
            if 0 <= target_idx < len(alive_ids):
                drone_idx = alive_ids[target_idx] if target_idx < len(alive_ids) else -1
                for d in alive_drones:
                    if d.id == drone_idx:
                        target_drone = d
                        break

            if target_drone is not None and target_drone.is_alive:
                engaged = self._check_engagement(
                    self.interceptor_positions[i], target_drone.position
                )
                if engaged:
                    target_drone.health -= 1.0
                    if target_drone.health <= 0:
                        target_drone.is_alive = False
                        target_drone.health = 0.0
                        self.neutralized_count += 1
                        commander_reward += self.CMD_NEUTRALIZE
                        self.engagement_log.append(
                            {
                                "step": self.current_step,
                                "interceptor": i,
                                "target": target_drone.id,
                            }
                        )
                    interceptor_rewards[i] += self.INT_HIT
                else:
                    interceptor_rewards[i] += self.INT_MISS
            else:
                interceptor_rewards[i] += self.INT_OUT_OF_ZONE

            interceptor_rewards[i] += self.INT_TIME

        # Cooperative bonus: 2+ interceptors on same target
        assignment_counts: Dict[int, int] = {}
        for a in self.commander_assignments:
            assignment_counts[int(a)] = assignment_counts.get(int(a), 0) + 1
        for i in range(self.n_interceptors):
            t = int(self.commander_assignments[i])
            if assignment_counts.get(t, 0) >= 2:
                interceptor_rewards[i] += self.INT_COOP_BONUS

        # Move enemy drones toward interceptors
        self._move_enemy_drones()

        # Check if any enemy drones reached the defended zone
        for drone in self.swarm.alive_drones:
            if np.linalg.norm(drone.position[:2]) < 50.0:
                commander_reward += self.CMD_MISSED

        # Check termination
        terminated = self.swarm.n_alive == 0
        truncated = self.current_step >= self.max_timesteps

        if terminated:
            commander_reward += self.CMD_CLEAR_BONUS

        obs = self._get_observations()
        rewards = {
            f"interceptor_{i}": float(interceptor_rewards[i]) for i in range(self.n_interceptors)
        }
        rewards["commander"] = float(commander_reward)

        info = {
            "n_alive_drones": self.swarm.n_alive,
            "neutralized": self.neutralized_count,
            "friendly_fire": self.friendly_fire_count,
            "step": self.current_step,
        }

        return obs, rewards, terminated, truncated, info

    def _move_enemy_drones(self) -> None:
        """Move enemy drones toward interceptors/defended zone."""
        if self.swarm is None:
            return
        defended_center = np.array([50.0, 50.0, 20.0], dtype=np.float32)
        for drone in self.swarm.alive_drones:
            speed = {"kamikaze": 40.0, "isr": 15.0, "jammer": 8.0}[drone.type]
            direction = defended_center - drone.position
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            noise = np.random.normal(0, 2.0, size=3).astype(np.float32)
            drone.velocity = (direction * speed + noise).astype(np.float32)
            step_scale = 0.05  # scale down for timestep
            drone.position = np.clip(
                drone.position + drone.velocity * step_scale, 0.0, float(self.grid_size)
            ).astype(np.float32)

    def _get_observations(self) -> Dict[str, Any]:
        """Build observation dict for all agents."""
        obs: Dict[str, Any] = {}

        # GAT observations for interceptors
        gat_obs = self._compute_gat_observations()

        for i in range(self.n_interceptors):
            radar = self._get_radar_returns(i)
            rf_spec = self._get_rf_spectrogram()
            own_state = np.concatenate(
                [
                    self.interceptor_positions[i],
                    self.interceptor_velocities[i],
                ]
            )
            task = int(self.commander_assignments[i]) if i < len(self.commander_assignments) else 0

            obs[f"interceptor_{i}"] = {
                "radar_returns": radar.astype(np.float32),
                "gat_neighbor_obs": gat_obs[i].astype(np.float32),
                "rf_spectrogram": rf_spec.astype(np.float32),
                "own_state": own_state.astype(np.float32),
                "commander_task": task,
            }

        # Commander observation
        swarm_state = (
            self.swarm.get_state_vector()
            if self.swarm
            else np.zeros(self.n_enemy_drones * 12, dtype=np.float32)
        )
        int_states = (
            self.interceptor_positions.flatten()
            if self.interceptor_positions is not None
            else np.zeros(self.n_interceptors * 6, dtype=np.float32)
        )
        int_vel_flat = (
            self.interceptor_velocities.flatten()
            if self.interceptor_velocities is not None
            else np.zeros(self.n_interceptors * 3, dtype=np.float32)
        )
        obs["commander"] = {
            "swarm_state": swarm_state.astype(np.float32),
            "interceptor_states": np.concatenate([int_states, int_vel_flat]).astype(np.float32)[
                : self.n_interceptors * 6
            ],
        }

        return obs

    def _get_radar_returns(self, interceptor_idx: int) -> np.ndarray:
        """Build radar return array for given interceptor."""
        returns = np.zeros((self.n_targets, 4), dtype=np.float32)
        pos = self.interceptor_positions[interceptor_idx]

        for t_idx, drone in enumerate(self.swarm.drones):
            if t_idx >= self.n_targets:
                break
            if not drone.is_alive:
                continue
            diff = drone.position - pos
            dist = float(np.linalg.norm(diff))
            if dist > self.radar_range:
                continue
            bearing = float(np.arctan2(diff[1], diff[0]))
            doppler = float(np.dot(drone.velocity, diff / (dist + 1e-8)))
            snr = max(0.0, 20.0 * (1.0 - dist / self.radar_range))

            raw = np.array([dist, bearing, doppler, snr], dtype=np.float32)

            # Apply EW jamming effects
            max_jam_effect = 0.0
            for hw_jammer in self.hw_jammers:
                effect = hw_jammer.compute_jamming_effect(pos)
                max_jam_effect = max(max_jam_effect, effect)

            if max_jam_effect > 0:
                raw = hw_jammer.corrupt_radar_return(raw, max_jam_effect)

            returns[t_idx] = raw

        return returns

    def _get_rf_spectrogram(self) -> np.ndarray:
        """Generate synthetic RF spectrogram from drone emissions."""
        spectrogram = np.zeros((64, 64), dtype=np.float32)
        if self.swarm is None:
            return spectrogram
        for drone in self.swarm.alive_drones:
            freq_bin = int(np.clip(drone.rf_emission_power * 30, 0, 63))
            time_bin = int(np.clip(self.current_step % 64, 0, 63))
            spectrogram[freq_bin, time_bin] += drone.rf_emission_power * 0.1
        spectrogram = np.clip(spectrogram, 0.0, 1.0)
        return spectrogram

    def _compute_gat_observations(self) -> np.ndarray:
        """Compute GAT-based neighbor observations for all interceptors."""
        gat_obs = np.zeros((self.n_interceptors, self.n_interceptors - 1, 64), dtype=np.float32)
        if self.gat_model is None or self.n_interceptors < 2:
            return gat_obs

        positions = self.interceptor_positions.copy()
        jammer_mask = np.array(
            [
                any(j.is_in_range(positions[i]) for j in self.hw_jammers)
                for i in range(self.n_interceptors)
            ]
        )

        edge_index, edge_attr = self.gat_model.build_adjacency(
            positions, self.comm_radius, jammer_mask
        )

        # Create node features from interceptor states
        node_features = np.concatenate(
            [
                self.interceptor_positions,
                self.interceptor_velocities,
            ],
            axis=-1,
        )  # (n_interceptors, 6)

        # Pad to gat input dim (64)
        node_feat_tensor = torch.zeros(self.n_interceptors, 64)
        node_feat_tensor[:, :6] = torch.tensor(node_features, dtype=torch.float32)

        with torch.no_grad():
            embeddings, _ = self.gat_model(node_feat_tensor, edge_index, edge_attr)

        embeddings_np = embeddings.numpy()  # (n_interceptors, 64)

        for i in range(self.n_interceptors):
            neighbors = [j for j in range(self.n_interceptors) if j != i]
            for k, nb in enumerate(neighbors):
                if k < self.n_interceptors - 1:
                    gat_obs[i, k] = embeddings_np[nb]

        return gat_obs

    def _apply_ew_effects(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Apply electronic warfare effects to observations."""
        for i in range(self.n_interceptors):
            key = f"interceptor_{i}"
            if key not in observations:
                continue
            pos = self.interceptor_positions[i]
            for hw_jammer in self.hw_jammers:
                effect = hw_jammer.compute_jamming_effect(pos)
                if effect > 0:
                    observations[key]["radar_returns"] = np.array(
                        [
                            hw_jammer.corrupt_radar_return(
                                observations[key]["radar_returns"][t], effect
                            )
                            for t in range(self.n_targets)
                        ]
                    )
                    if self._in_gps_denial_zone(pos):
                        observations[key]["own_state"][:3] += np.random.normal(0, 0.5, 3).astype(
                            np.float32
                        )
        return observations

    def _check_engagement(self, interceptor_pos: np.ndarray, target_pos: np.ndarray) -> bool:
        """Check if interceptor is in engagement range of target."""
        dist = float(np.linalg.norm(interceptor_pos - target_pos))
        return dist <= 30.0

    def _in_gps_denial_zone(self, pos: np.ndarray) -> bool:
        """Check if position is in any GPS denial zone."""
        for center in self.gps_denial_centers:
            if np.linalg.norm(pos[:2] - center[:2]) < 80.0:
                return True
        return False

    def _build_swarm_graph(self):
        """Build a torch_geometric Data object from current swarm state."""
        from torch_geometric.data import Data

        alive = self.swarm.alive_drones if self.swarm else []
        if not alive:
            return Data(
                x=torch.zeros(1, 12),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
            )

        positions = np.stack([d.position for d in alive])
        velocities = np.stack([d.velocity for d in alive])
        type_enc = np.array([{"kamikaze": 0, "isr": 1, "jammer": 2}[d.type] for d in alive])
        health = np.array([d.health for d in alive])
        node_feats = np.concatenate(
            [positions, velocities, health[:, None], type_enc[:, None]], axis=-1
        )
        x = torch.tensor(node_feats, dtype=torch.float32)

        # Build edge index (drones within 100m)
        n = len(alive)
        edges_src, edges_dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j and np.linalg.norm(positions[i] - positions[j]) < 100.0:
                    edges_src.append(i)
                    edges_dst.append(j)

        if edges_src:
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    def render(self, mode: str = "human"):
        """Render the current environment state as a matplotlib figure."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#0f0f23")

        # Terrain overlay
        if self.terrain is not None:
            terrain_data = self.terrain.get_contour_data()
            ax.contourf(
                terrain_data.T,
                levels=10,
                cmap="terrain",
                alpha=0.3,
                extent=[0, self.grid_size, 0, self.grid_size],
            )

        # Jammer zones
        for hw_jammer in self.hw_jammers:
            circle = patches.Circle(
                hw_jammer.position[:2],
                hw_jammer.radius,
                color="red",
                alpha=0.2,
                linewidth=2,
                linestyle="--",
                fill=True,
            )
            ax.add_patch(circle)
            ax.add_patch(
                patches.Circle(
                    hw_jammer.position[:2],
                    hw_jammer.radius,
                    color="red",
                    alpha=0.6,
                    fill=False,
                    linewidth=2,
                )
            )

        # GPS denial zones
        for center in self.gps_denial_centers:
            circle = patches.Circle(center[:2], 80, color="gray", alpha=0.15, hatch="//", fill=True)
            ax.add_patch(circle)

        # Radar coverage arcs
        for i in range(self.n_interceptors):
            if self.interceptor_positions is not None:
                pos = self.interceptor_positions[i]
                arc = patches.Circle(
                    pos[:2],
                    self.radar_range,
                    color="green",
                    alpha=0.1,
                    fill=False,
                    linewidth=1,
                    linestyle="--",
                )
                ax.add_patch(arc)

        # Enemy drones
        type_colors = {"kamikaze": "red", "isr": "orange", "jammer": "purple"}
        if self.swarm:
            for drone in self.swarm.drones:
                color = type_colors.get(drone.type, "red")
                alpha = 1.0 if drone.is_alive else 0.3
                marker = "x" if not drone.is_alive else "o"
                ax.scatter(
                    drone.position[0],
                    drone.position[1],
                    c=color,
                    s=80,
                    marker=marker,
                    alpha=alpha,
                    zorder=5,
                )
                if drone.is_alive:
                    # Health bar
                    ax.annotate(
                        f"{drone.health:.1f}",
                        (drone.position[0], drone.position[1] + 8),
                        ha="center",
                        fontsize=6,
                        color=color,
                    )

        # Interceptors
        if self.interceptor_positions is not None:
            for i in range(self.n_interceptors):
                pos = self.interceptor_positions[i]
                vel = (
                    self.interceptor_velocities[i]
                    if self.interceptor_velocities is not None
                    else np.zeros(3)
                )
                ax.scatter(pos[0], pos[1], c="cyan", s=120, marker="^", zorder=6)
                if np.linalg.norm(vel) > 1e-3:
                    ax.annotate(
                        "",
                        xy=(pos[0] + vel[0] * 0.5, pos[1] + vel[1] * 0.5),
                        xytext=(pos[0], pos[1]),
                        arrowprops=dict(arrowstyle="->", color="cyan", lw=1.5),
                    )

        # Active engagements
        if self.swarm and self.interceptor_positions is not None:
            alive_drones = self.swarm.alive_drones
            alive_ids = {d.id: d for d in alive_drones}
            for i in range(self.n_interceptors):
                t = int(self.commander_assignments[i]) if i < len(self.commander_assignments) else 0
                alive_list = list(alive_ids.values())
                if 0 <= t < len(alive_list):
                    td = alive_list[t]
                    if td.is_alive:
                        int_pos = self.interceptor_positions[i]
                        ax.plot(
                            [int_pos[0], td.position[0]],
                            [int_pos[1], td.position[1]],
                            "y-",
                            alpha=0.7,
                            lw=1.5,
                        )

        # Score overlay
        n_alive = self.swarm.n_alive if self.swarm else 0
        time_remaining = self.max_timesteps - self.current_step
        score_text = (
            f"Neutralized: {self.neutralized_count}/{self.n_enemy_drones}\n"
            f"Alive: {n_alive}\n"
            f"Step: {self.current_step}/{self.max_timesteps}\n"
            f"Time Left: {time_remaining}"
        )
        ax.text(
            0.98,
            0.98,
            score_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="white",
            bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.8),
        )

        ax.set_title("SWARM-SHIELD: C-UAS Environment", color="white", fontsize=12)
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")

        plt.tight_layout()

        if mode == "rgb_array":
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(height, width, 3)
            plt.close(fig)
            return image

        return fig

    def close(self) -> None:
        """Close rendering resources."""
        import matplotlib.pyplot as plt

        plt.close("all")


if __name__ == "__main__":
    env = CUASEnv()
    obs, info = env.reset(seed=42)
    print("Observation keys:", list(obs.keys()))
    print("Interceptor 0 radar shape:", obs["interceptor_0"]["radar_returns"].shape)

    # Take a random step
    actions = {f"interceptor_{i}": np.random.uniform(-1, 1, 3) for i in range(4)}
    actions["commander"] = np.array([0, 1, 2, 3])
    obs2, rewards, terminated, truncated, info = env.step(actions)
    print("Rewards:", rewards)
    print("Info:", info)
    print("cuas_env.py OK")
