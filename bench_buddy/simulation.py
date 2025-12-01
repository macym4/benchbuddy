"""High-level Bench Buddy simulation orchestrator.

The implementation is intentionally kinematic: we prescribe Cartesian
trajectories for the human arms / bar and monitor safety constraints.
MeshCat is used for visualization while Matplotlib handles quick plots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import (
    BarWaypoint,
    FailureEvent,
    RandomnessConfig,
    SensorReadings,
    SimulationConfig,
    SimulationResult,
)

# pydrake imports are intentionally local so that unit tests that do not have
# Drake installed can still import this module to inspect dataclasses.
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    RigidTransform,
    RotationMatrix,
    StartMeshcat,
)
from pydrake.multibody.tree import Joint


def _convert_to_np(vec: Sequence[float]) -> np.ndarray:
    """Convert to a numpy vector with shape (3,)."""
    arr = np.asarray(vec, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"Expected 3D vector, received shape {arr.shape}")
    return arr


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class _TrajectoryBundle:
    times: List[float]
    left_positions: List[np.ndarray]
    right_positions: List[np.ndarray]
    bar_positions: List[np.ndarray]
    robot_positions: List[np.ndarray]
    bar_velocities: List[np.ndarray]
    robot_velocities: List[np.ndarray]
    failure: FailureEvent | None


class BenchBuddySimulation:
    """Orchestrates the full Bench Buddy scenario."""

    # Reference poses in the world frame (meters).
    _BENCH_IN_WORLD = np.array([-0.15, 0.0, 0.127])
    _TORSO_IN_WORLD = np.array([-0.45, 0.0, 0.697])
    _NECK_IN_WORLD = np.array([-0.15, 0.0, 0.867])
    _HEAD_IN_WORLD = np.array([0.15, 0.0, 0.987])
    _LEFT_ARM_HOME = np.array([-0.45, 0.25, 0.697])
    _RIGHT_ARM_HOME = np.array([-0.45, -0.25, 0.697])
    _BAR_RACK_HOME = np.array([-0.06, 0.0, 0.855])
    _RACK_HOME = np.array([-0.15, 0.0, 0.127])
    _HAND_OFFSET_IN_ARM = np.array([0.30, 0.0, 0.0])
    _PR2_IDLE = {
        "l_shoulder_pan_joint": 0.6,
        "l_shoulder_lift_joint": 0.2,
        "l_upper_arm_roll_joint": 0.0,
        "l_elbow_flex_joint": -0.5,
        "l_forearm_roll_joint": 0.0,
        "l_wrist_flex_joint": -0.2,
        "l_wrist_roll_joint": 0.0,
        "r_shoulder_pan_joint": -0.6,
        "r_shoulder_lift_joint": 0.2,
        "r_upper_arm_roll_joint": 0.0,
        "r_elbow_flex_joint": -0.5,
        "r_forearm_roll_joint": 0.0,
        "r_wrist_flex_joint": -0.2,
        "r_wrist_roll_joint": 0.0,
    }
    _PR2_RESCUE = {
        "l_shoulder_pan_joint": 0.0,
        "l_shoulder_lift_joint": 1.1,
        "l_upper_arm_roll_joint": 0.0,
        "l_elbow_flex_joint": -1.2,
        "l_forearm_roll_joint": 0.0,
        "l_wrist_flex_joint": -0.6,
        "l_wrist_roll_joint": 0.0,
        "r_shoulder_pan_joint": 0.0,
        "r_shoulder_lift_joint": 1.1,
        "r_upper_arm_roll_joint": 0.0,
        "r_elbow_flex_joint": -1.2,
        "r_forearm_roll_joint": 0.0,
        "r_wrist_flex_joint": -0.6,
        "r_wrist_roll_joint": 0.0,
    }

    def __init__(self, config: SimulationConfig):
        """Standard building of model. Starts the Meschat, makes the builder,
        adds to it, creates context and adds to it, finalizes the plant, etc"""
        self.config = config
        self.assets_dir = Path(__file__).resolve().parent.parent / "assets"
        if config.meshcat is None:
            self.meshcat = StartMeshcat()
        else:
            self.meshcat = config.meshcat

        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=0.0
        )
        self.parser = Parser(self.plant, self.scene_graph)

        # Optional PR2 joints are populated while loading the environment.
        self.pr2_joints: Dict[str, Joint] = {}
        self.pr2_model = None

        self._load_environment()
        self.plant.Finalize()

        MeshcatVisualizer.AddToBuilder(
            self.builder,
            self.scene_graph,
            self.meshcat,
            MeshcatVisualizerParams(),
        )
        self.diagram = self.builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)

        self.noise = config.randomness or RandomnessConfig()
        self._rng = np.random.default_rng(self.noise.seed)

        self._safe_upper_z = self._NECK_IN_WORLD[2] + 0.3048
        self._safe_lower_z = self._BAR_RACK_HOME[2] - 0.3048

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> SimulationResult:
        """Execute a full simulation, record MeshCat animation, and return logs."""

        traj = self._synthesize_trajectory()
        self._record_meshcat_animation(traj)

        sensors = self._synthesize_sensors(traj)
        result = SimulationResult(
            times=traj.times,
            bar_positions=[tuple(p.tolist()) for p in traj.bar_positions],
            bar_velocities=[tuple(v.tolist()) for v in traj.bar_velocities],
            robot_positions=[tuple(p.tolist()) for p in traj.robot_positions],
            robot_velocities=[tuple(v.tolist()) for v in traj.robot_velocities],
            failure_events=[traj.failure] if traj.failure else [],
            sensors=sensors,
        )
        return result

    # ------------------------------------------------------------------ #
    # Diagram / world construction
    # ------------------------------------------------------------------ #
    def _load_environment(self) -> None:
        wf = self.plant.world_frame()
        bench_model = self.parser.AddModels(str(self.assets_dir / "bench.sdf"))[0]
        bench_frame = self.plant.GetFrameByName("bench", bench_model)
        self.plant.WeldFrames(
            wf,
            bench_frame,
            RigidTransform(p=self._BENCH_IN_WORLD),
        )

        torso_model = self.parser.AddModels(str(self.assets_dir / "torso.sdf"))[0]
        neck_model = self.parser.AddModels(str(self.assets_dir / "neck.sdf"))[0]
        head_model = self.parser.AddModels(str(self.assets_dir / "head.sdf"))[0]

        torso_frame = self.plant.GetFrameByName("torso", torso_model)
        neck_frame = self.plant.GetFrameByName("neck", neck_model)
        head_frame = self.plant.GetFrameByName("head", head_model)

        self.plant.WeldFrames(
            bench_frame,
            torso_frame,
            RigidTransform(p=self._TORSO_IN_WORLD - self._BENCH_IN_WORLD),
        )
        self.plant.WeldFrames(
            torso_frame,
            neck_frame,
            RigidTransform(p=self._NECK_IN_WORLD - self._TORSO_IN_WORLD),
        )
        self.plant.WeldFrames(
            neck_frame,
            head_frame,
            RigidTransform(p=self._HEAD_IN_WORLD - self._NECK_IN_WORLD),
        )

        rack_model = self.parser.AddModels(str(self.assets_dir / "rack.sdf"))[0]
        rack_frame = self.plant.GetFrameByName("rack", rack_model)
        self.plant.WeldFrames(
            wf,
            rack_frame,
            RigidTransform(p=self._RACK_HOME),
        )

        # Floating bodies that we'll  re-position at sample time.
        self.left_arm_model = self.parser.AddModels(
            str(self.assets_dir / "left_arm.sdf")
        )[0]
        self.right_arm_model = self.parser.AddModels(
            str(self.assets_dir / "right_arm.sdf")
        )[0]
        self.left_hand_model = self.parser.AddModels(
            str(self.assets_dir / "left_hand.sdf")
        )[0]
        self.right_hand_model = self.parser.AddModels(
            str(self.assets_dir / "right_hand.sdf")
        )[0]
        self.bar_model = self.parser.AddModels(str(self.assets_dir / "bar.sdf"))[0]

        self.left_arm_body = self.plant.GetBodyByName("left_arm", self.left_arm_model)
        self.right_arm_body = self.plant.GetBodyByName("right_arm", self.right_arm_model)
        self.left_hand_body = self.plant.GetBodyByName("left_hand", self.left_hand_model)
        self.right_hand_body = self.plant.GetBodyByName(
            "right_hand", self.right_hand_model
        )
        self.bar_body = self.plant.GetBodyByName("bar", self.bar_model)

        # Optional PR2 (for visualization only). Disable by default because
        # the stock PR2 URDF contains mimic joints that introduce kinematic
        # loops when simulated in a continuous MultibodyPlant.
        if self.config.include_pr2:
            try:
                self.pr2_model = self.parser.AddModelsFromUrl(
                    "package://drake_models/pr2_description/urdf/pr2_simplified.urdf"
                )[0]
                base_frame = self.plant.GetFrameByName("base_footprint", self.pr2_model)
                self.plant.WeldFrames(
                    wf,
                    base_frame,
                    RigidTransform(p=[-1.2, 0.0, 0.0]),
                )
                for joint_name in self._PR2_IDLE.keys():
                    try:
                        joint = self.plant.GetJointByName(joint_name, self.pr2_model)
                        self.pr2_joints[joint_name] = joint
                    except RuntimeError:
                        continue
            except RuntimeError:
                self.pr2_model = None

    # ------------------------------------------------------------------ #
    # Trajectory synthesis
    # ------------------------------------------------------------------ #
    def _synthesize_trajectory(self) -> _TrajectoryBundle:
        dt = 1.0 / self.config.sample_rate_hz

        # Ensure we start at rest on the rack.
        start_offset = np.zeros(3)
        waypoints = [BarWaypoint("rack", start_offset, hold_time=0.25)]
        waypoints.extend(self.config.waypoints)
        waypoints.append(BarWaypoint("rack_return", start_offset, hold_time=0.25))

        left_positions: List[np.ndarray] = []
        right_positions: List[np.ndarray] = []
        bar_positions: List[np.ndarray] = []
        robot_positions: List[np.ndarray] = []
        times: List[float] = []
        bar_velocities: List[np.ndarray] = []
        robot_velocities: List[np.ndarray] = []

        t = 0.0
        last_left = self._LEFT_ARM_HOME.copy()
        last_right = self._RIGHT_ARM_HOME.copy()
        last_bar = self._BAR_RACK_HOME.copy()
        last_robot = np.array([-0.8, 0.0, 0.95])

        failure: FailureEvent | None = None
        rescue_in_progress = False
        rescue_target = self._NECK_IN_WORLD + np.array([0.0, 0.0, 0.35])

        for idx, wp in enumerate(waypoints[:-1]):
            next_wp = waypoints[idx + 1]
            rel_a = _convert_to_np(wp.relative_position)
            rel_b = _convert_to_np(next_wp.relative_position)
            world_a_left = self._LEFT_ARM_HOME + rel_a
            world_b_left = self._LEFT_ARM_HOME + rel_b
            world_a_right = self._RIGHT_ARM_HOME + rel_a
            world_b_right = self._RIGHT_ARM_HOME + rel_b

            segment_length = np.linalg.norm(world_b_left - world_a_left)
            duration = segment_length / max(self.config.peak_bar_velocity, 1e-3)
            duration = max(duration, 0.05)
            samples = max(1, int(math.ceil(duration / dt)))

            for step in range(samples):
                alpha = (step + 1) / samples
                left = (1 - alpha) * world_a_left + alpha * world_b_left
                right = (1 - alpha) * world_a_right + alpha * world_b_right

                if self.noise.enable:
                    left = self._inject_noise(left)
                    right = self._inject_noise(right)

                bar = self._bar_from_arms(left, right)
                robot_target = last_robot.copy()

                if not failure:
                    failure_reason = self._check_failure(bar, last_bar, dt)
                    if failure_reason:
                        failure = FailureEvent(time=t, reason=failure_reason, bar_position=tuple(bar))
                        rescue_in_progress = True

                if rescue_in_progress:
                    robot_target = self._robot_rescue_pose(bar, t, failure.time if failure else t)
                    bar = robot_target + np.array([0.0, 0.0, 0.05])
                    left = self._left_from_bar(bar)
                    right = self._right_from_bar(bar)

                t += dt
                left_positions.append(left)
                right_positions.append(right)
                bar_positions.append(bar)
                robot_positions.append(robot_target)
                times.append(t)

                bar_velocities.append((bar - last_bar) / dt)
                robot_velocities.append((robot_target - last_robot) / dt)
                last_left = left
                last_right = right
                last_bar = bar
                last_robot = robot_target

            # dwell
            dwell_samples = max(1, int(math.ceil(next_wp.hold_time / dt)))
            for _ in range(dwell_samples):
                t += dt
                left_positions.append(last_left.copy())
                right_positions.append(last_right.copy())
                bar_positions.append(last_bar.copy())
                robot_positions.append(last_robot.copy())
                times.append(t)
                bar_velocities.append(np.zeros(3))
                robot_velocities.append(np.zeros(3))

        return _TrajectoryBundle(
            times=times,
            left_positions=left_positions,
            right_positions=right_positions,
            bar_positions=bar_positions,
            robot_positions=robot_positions,
            bar_velocities=bar_velocities,
            robot_velocities=robot_velocities,
            failure=failure,
        )

    def _inject_noise(self, vec: np.ndarray) -> np.ndarray:
        """Clamp Gaussian noise to remain within safe ranges."""
        noisy = vec + self._rng.normal(0.0, self.noise.position_sigma, size=3)
        noisy[2] = _clamp(noisy[2], self._safe_lower_z + 0.01, self._safe_upper_z - 0.01)
        return noisy

    def _bar_from_arms(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        midpoint = 0.5 * (left + right)
        return midpoint + np.array([0.09, 0.0, 0.02])

    def _left_from_bar(self, bar: np.ndarray) -> np.ndarray:
        return bar - np.array([0.09, 0.0, 0.02]) + np.array([0.0, 0.10, -0.02])

    def _right_from_bar(self, bar: np.ndarray) -> np.ndarray:
        return bar - np.array([0.09, 0.0, 0.02]) + np.array([0.0, -0.10, -0.02])

    def _check_failure(self, bar: np.ndarray, prev_bar: np.ndarray, dt: float) -> str | None:
        vel = np.linalg.norm((bar - prev_bar) / max(dt, 1e-6))
        if vel > 1.0:
            return "velocity_exceeded_1mps"
        centered = np.linalg.norm(bar[:2] - self._NECK_IN_WORLD[:2]) < 0.12
        if centered and self._NECK_IN_WORLD[2] < bar[2] < self._safe_upper_z:
            return "entered_neck_guard_zone"
        if bar[2] < self._safe_lower_z:
            return "dropped_toward_stomach"
        return None

    def _robot_rescue_pose(self, bar: np.ndarray, t: float, failure_time: float) -> np.ndarray:
        ramp = _clamp((t - failure_time) / 0.35, 0.0, 1.0)
        travel = self._NECK_IN_WORLD + np.array([-0.05, 0.0, 0.28])
        catch_point = bar + np.array([-0.05, 0.0, -0.10])
        return (1 - ramp) * travel + ramp * catch_point

    # ------------------------------------------------------------------ #
    # MeshCat animation
    # ------------------------------------------------------------------ #
    def _record_meshcat_animation(self, traj: _TrajectoryBundle) -> None:
        self.meshcat.Delete()
        self.meshcat.DeleteAddedControls()
        self.meshcat.StartRecording(set_visualizations_while_recording=False)
        bar_rotation = RotationMatrix.MakeXRotation(math.pi / 2.0)
        arm_rotation = RotationMatrix.MakeZRotation(math.pi / 2.0)
        hand_offset_world = arm_rotation.multiply(self._HAND_OFFSET_IN_ARM)

        for t, left, right, bar in zip(
            traj.times, traj.left_positions, traj.right_positions, traj.bar_positions
        ):
            self.context.SetTime(t)
            self.plant.SetFreeBodyPose(
                self.plant_context,
                self.left_arm_body,
                RigidTransform(R=arm_rotation, p=left),
            )
            self.plant.SetFreeBodyPose(
                self.plant_context,
                self.left_hand_body,
                RigidTransform(R=arm_rotation, p=left + hand_offset_world),
            )
            self.plant.SetFreeBodyPose(
                self.plant_context,
                self.right_arm_body,
                RigidTransform(R=arm_rotation, p=right),
            )
            self.plant.SetFreeBodyPose(
                self.plant_context,
                self.right_hand_body,
                RigidTransform(R=arm_rotation, p=right + hand_offset_world),
            )
            self.plant.SetFreeBodyPose(
                self.plant_context,
                self.bar_body,
                RigidTransform(R=bar_rotation, p=bar + np.array([0.0, 0.0, 0.5])),
            )
            self._apply_pr2_pose(traj.failure, t)
            self.diagram.ForcedPublish(self.context)

        self.meshcat.StopRecording()
        if self.config.show_meshcat:
            self.meshcat.PublishRecording()

    def _apply_pr2_pose(self, failure: FailureEvent | None, time: float) -> None:
        if not self.pr2_joints:
            return
        rescue_active = failure is not None and time >= failure.time
        target = self._PR2_RESCUE if rescue_active else self._PR2_IDLE
        for name, angle in target.items():
            joint = self.pr2_joints.get(name)
            if joint is None:
                continue
            try:
                joint.set_angle(self.plant_context, angle)
            except AttributeError:
                start = joint.position_start()
                count = joint.num_positions()
                model = joint.model_instance()
                q = self.plant.GetPositions(self.plant_context, model)
                q_slice = q.copy()
                q_slice[start : start + count] = angle
                self.plant.SetPositions(self.plant_context, model, q_slice)

    # ------------------------------------------------------------------ #
    # Sensors / plots
    # ------------------------------------------------------------------ #
    def _synthesize_sensors(self, traj: _TrajectoryBundle) -> SensorReadings:
        sensors = SensorReadings()
        vel_sigma = self.noise.velocity_sigma if self.noise.enable else 0.0

        for t, bar, rob, bar_vel, rob_vel in zip(
            traj.times,
            traj.bar_positions,
            traj.robot_positions,
            traj.bar_velocities,
            traj.robot_velocities,
        ):
            dist = np.linalg.norm(bar - self._NECK_IN_WORLD)
            depth_noise = (
                self._rng.normal(0.0, self.noise.perception_sigma)
                if self.noise.enable
                else 0.0
            )
            sensors.depth_measurements.append((t, dist + depth_noise))
            sensors.rgb_debug_stream.append((t, tuple(bar.tolist())))
            noisy_joint = tuple((rob + self._rng.normal(0.0, vel_sigma, size=3)).tolist())
            sensors.joint_encoder_log.append((t, noisy_joint))
        return sensors

    # Convenience plotting API (imported lazily to keep notebook light)
    def plot(self, result: SimulationResult, show_bar: bool = True, show_robot: bool = True):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        times = result.times

        if show_bar:
            bar_pos = np.array(result.bar_positions)
            bar_vel = np.array(result.bar_velocities)
            axes[0].plot(times, bar_pos[:, 2], label="Bar z")
            axes[0].plot(times, bar_pos[:, 0], label="Bar x")
            axes[0].plot(times, bar_pos[:, 1], label="Bar y")
            axes[1].plot(times, np.linalg.norm(bar_vel, axis=1), label="|v_bar|")

        if show_robot:
            rob_pos = np.array(result.robot_positions)
            rob_vel = np.array(result.robot_velocities)
            axes[0].plot(times, rob_pos[:, 2], "--", label="Robot z")
            axes[1].plot(times, np.linalg.norm(rob_vel, axis=1), "--", label="|v_robot|")

        if result.failure_events:
            for event in result.failure_events:
                for ax in axes:
                    ax.axvline(event.time, color="r", linestyle=":", alpha=0.6)
                    ax.text(
                        event.time,
                        ax.get_ylim()[1],
                        event.reason,
                        rotation=90,
                        va="bottom",
                        fontsize=8,
                    )

        axes[0].set_ylabel("Position (m)")
        axes[1].set_ylabel("Velocity (m/s)")
        axes[1].set_xlabel("Time (s)")
        for ax in axes:
            ax.legend(loc="upper right")
            ax.grid(True)
        fig.tight_layout()
        return fig, axes
