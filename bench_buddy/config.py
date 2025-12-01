"""Dataclasses and typed containers for the Bench Buddy simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class BarWaypoint:
    """Single Cartesian waypoint for the lift trajectory."""

    name: str
    relative_position: Sequence[float]
    hold_time: float = 0.15  # seconds to pause at this waypoint


@dataclass
class RandomnessConfig:
    """Gaussian noise configuration for arm motion and sensing."""

    enable: bool = True
    position_sigma: float = 0.006  # meters
    velocity_sigma: float = 0.06  # m/s
    perception_sigma: float = 0.01  # meters
    seed: int = 13


@dataclass
class SimulationConfig:
    """High-level inputs that define a simulation rollout."""

    peak_bar_velocity: float
    waypoints: List[BarWaypoint]
    meshcat: Optional[object] = None
    sample_rate_hz: float = 200.0
    randomness: RandomnessConfig = field(default_factory=RandomnessConfig)
    publish_plots: bool = True
    show_meshcat: bool = True
    include_pr2: bool = False


@dataclass
class FailureEvent:
    """Bookkeeping for any failure and resulting intervention."""

    time: float
    reason: str
    bar_position: Tuple[float, float, float]


@dataclass
class SensorReadings:
    """Aggregated sensing logs used by the rescue logic."""

    depth_measurements: List[Tuple[float, float]] = field(default_factory=list)
    rgb_debug_stream: List[Tuple[float, Tuple[float, float, float]]] = field(
        default_factory=list
    )
    joint_encoder_log: List[Tuple[float, Tuple[float, ...]]] = field(
        default_factory=list
    )


@dataclass
class SimulationResult:
    """Bundle containing every useful signal from a rollout."""

    times: List[float]
    bar_positions: List[Tuple[float, float, float]]
    bar_velocities: List[Tuple[float, float, float]]
    robot_positions: List[Tuple[float, float, float]]
    robot_velocities: List[Tuple[float, float, float]]
    failure_events: List[FailureEvent]
    sensors: SensorReadings
