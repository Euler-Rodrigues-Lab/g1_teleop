# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""MuJoCo controller for G1 full-body teleoperation.

Standalone implementation extracted from the original controller.
Applies joint position goals to the simulation and manages the mocap body /
follow camera. `set_joint_goals` is duck-typed: it accepts either the legacy
goals dict (``{"q_goal_right": ..., ...}``) or any object exposing the same
names as attributes — in particular ``geo_kin_core.types.RetargetOutput``
(the licensed `geo_kin` wheel returns such objects). ``None`` for a joint
group means "keep the previous goal".
"""

import time
from collections import defaultdict
from collections.abc import Mapping
from typing import Optional

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# Joint groups consumed from a goals dict / RetargetOutput-like object.
_GOAL_KEYS = (
    "q_goal_right",
    "q_goal_left",
    "q_goal_right_leg",
    "q_goal_left_leg",
    "q_goal_right_hand",
    "q_goal_left_hand",
    "q_goal_torso",
)


def _get_goal(goals, key):
    """Read a joint-group goal from a dict or an attribute-style object."""
    if isinstance(goals, Mapping):
        return goals.get(key)
    return getattr(goals, key, None)


class G1FullBodyMuJoCoController:
    """MuJoCo controller for the G1: applies joint position goals + viz state."""

    def __init__(self, mujoco_model, mujoco_data, debug=False, timing_enabled=False,
                 singularity_overlay_enabled=False):
        """
        Args:
            mujoco_model: MuJoCo model.
            mujoco_data: MuJoCo data.
            debug: Enable debug output.
            timing_enabled: Enable performance timing measurements.
            singularity_overlay_enabled: Enable singularity overlay.
        """
        self.model = mujoco_model
        self.data = mujoco_data
        self.debug = debug
        self.timing_enabled = timing_enabled
        self.singularity_overlay_enabled = singularity_overlay_enabled

        # Timing statistics
        if self.timing_enabled:
            self.timing_stats = defaultdict(list)

        # Get joint indices in MuJoCo model
        self._setup_joint_indices()

        # Initial positions
        self._update_current_positions()

        # Store initial positions for drift prevention
        self._store_initial_positions()

        # Initialize goals to current positions
        self.q_goal_right = self.q_current_right.copy()
        self.q_goal_left = self.q_current_left.copy()
        self.q_goal_right_leg = self.q_current_right_leg.copy()
        self.q_goal_left_leg = self.q_current_left_leg.copy()
        self.q_goal_right_hand = self.q_current_right_hand.copy()
        self.q_goal_left_hand = self.q_current_left_hand.copy()
        self.q_goal_torso = self.q_current_torso.copy()

        # Mocap Body
        self.mocap_enabled = False
        self.mocap_body_name = "pelvis_mocap_mover"
        self.mocap_index = None
        self.quat_prev_mocap = None
        self.quat_filter_alpha = 0.1

        # Camera
        self.follow_camera_enabled = False

        if self.debug:
            print("G1 MuJoCo Controller initialized")

    def _setup_joint_indices(self):
        """Setup joint indices and names for all body parts."""
        self.right_arm_joint_names = [
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ]
        self.left_arm_joint_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"
        ]
        self.right_leg_joint_names = [
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
        ]
        self.left_leg_joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"
        ]
        self.right_hand_joint_names = [
            "index_mcp", "index_pip", "middle_mcp", "middle_pip",
            "ring_mcp", "ring_pip", "pinky_mcp", "pinky_pip",
            "thumb_mcp", "thumb_cmc"
        ]
        self.left_hand_joint_names = [
            "l_index_mcp", "l_index_pip", "l_middle_mcp", "l_middle_pip",
            "l_ring_mcp", "l_ring_pip", "l_pinky_mcp", "l_pinky_pip",
            "l_thumb_mcp", "l_thumb_cmc"
        ]
        self.torso_joint_names = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

        # Build name -> ID mapping
        joint_name2id = {}
        for i in range(self.model.njnt):
            name = self.model.joint(i).name
            if name:
                joint_name2id[name] = i

        def get_addrs(names, type='qpos'):
            addrs = []
            for name in names:
                if name in joint_name2id:
                    jid = joint_name2id[name]
                    if type == 'qpos':
                        addrs.append(self.model.joint(jid).qposadr[0])
                    else:
                        addrs.append(self.model.joint(jid).dofadr[0])
            return addrs

        # Qpos addresses
        self.right_arm_qpos_addrs = get_addrs(self.right_arm_joint_names, 'qpos')
        self.left_arm_qpos_addrs = get_addrs(self.left_arm_joint_names, 'qpos')
        self.right_leg_qpos_addrs = get_addrs(self.right_leg_joint_names, 'qpos')
        self.left_leg_qpos_addrs = get_addrs(self.left_leg_joint_names, 'qpos')
        self.right_hand_qpos_addrs = get_addrs(self.right_hand_joint_names, 'qpos')
        self.left_hand_qpos_addrs = get_addrs(self.left_hand_joint_names, 'qpos')
        self.torso_qpos_addrs = get_addrs(self.torso_joint_names, 'qpos')

        # Actuator indices
        self.right_arm_actuator_ids = []
        self.left_arm_actuator_ids = []
        self.right_leg_actuator_ids = []
        self.left_leg_actuator_ids = []
        self.right_hand_actuator_ids = []
        self.left_hand_actuator_ids = []
        self.torso_actuator_ids = []
        self.other_actuator_ids = []

        controlled_names = set(self.right_arm_joint_names + self.left_arm_joint_names +
                               self.right_leg_joint_names + self.left_leg_joint_names +
                               self.right_hand_joint_names + self.left_hand_joint_names +
                               self.torso_joint_names)

        for i in range(self.model.nu):
            name = self.model.actuator(i).name
            if name in self.right_arm_joint_names:
                self.right_arm_actuator_ids.append(i)
            elif name in self.left_arm_joint_names:
                self.left_arm_actuator_ids.append(i)
            elif name in self.right_leg_joint_names:
                self.right_leg_actuator_ids.append(i)
            elif name in self.left_leg_joint_names:
                self.left_leg_actuator_ids.append(i)
            elif name in self.right_hand_joint_names:
                self.right_hand_actuator_ids.append(i)
            elif name in self.left_hand_joint_names:
                self.left_hand_actuator_ids.append(i)
            elif name in self.torso_joint_names:
                self.torso_actuator_ids.append(i)
            elif name not in controlled_names:
                self.other_actuator_ids.append(i)

    def _update_current_positions(self):
        """Update current joint positions from MuJoCo data."""
        self.q_current_right = np.array([self.data.qpos[i] for i in self.right_arm_qpos_addrs])
        self.q_current_left = np.array([self.data.qpos[i] for i in self.left_arm_qpos_addrs])
        self.q_current_right_leg = np.array([self.data.qpos[i] for i in self.right_leg_qpos_addrs])
        self.q_current_left_leg = np.array([self.data.qpos[i] for i in self.left_leg_qpos_addrs])
        self.q_current_right_hand = np.array([self.data.qpos[i] for i in self.right_hand_qpos_addrs])
        self.q_current_left_hand = np.array([self.data.qpos[i] for i in self.left_hand_qpos_addrs])
        self.q_current_torso = np.array([self.data.qpos[i] for i in self.torso_qpos_addrs])

    def _store_initial_positions(self):
        """Store initial positions of other (non-teleoperated) actuators."""
        self.q_initial_other = {}
        for aid in self.other_actuator_ids:
            self.q_initial_other[aid] = self.data.ctrl[aid]

    def set_joint_goals(self, goals):
        """Set target joint angles.

        Args:
            goals: A dict with ``q_goal_*`` keys (legacy monolith contract) OR
                any object exposing the same names as attributes — e.g.
                ``geo_kin_core.types.RetargetOutput`` (dataclass from the
                reference backend) or the object returned by the licensed
                ``geo_kin`` wheel. ``None`` (or a missing key/attribute) means
                "keep previous goal" for that joint group.
        """
        if goals is None:
            return
        if isinstance(goals, Mapping) and not goals:
            return

        for key in _GOAL_KEYS:
            value = _get_goal(goals, key)
            if value is not None:
                setattr(self, key, np.asarray(value, dtype=float))

    def apply_control(self, engaged=True, kinematic_mode=False):
        """Apply control commands (position or kinematic) to the robot."""
        self._update_current_positions()

        if not engaged:
            # Hold: keep applying the last goals (positions do not move).
            pass

        if kinematic_mode:
            self._apply_kinematic_control()
        else:
            self._apply_position_control()

    def _apply_position_control(self):
        def apply(actuators, goals):
            for i, aid in enumerate(actuators):
                if i < len(goals):
                    self.data.ctrl[aid] = goals[i]

        apply(self.right_arm_actuator_ids, self.q_goal_right)
        apply(self.left_arm_actuator_ids, self.q_goal_left)
        apply(self.right_leg_actuator_ids, self.q_goal_right_leg)
        apply(self.left_leg_actuator_ids, self.q_goal_left_leg)
        apply(self.right_hand_actuator_ids, self.q_goal_right_hand)
        apply(self.left_hand_actuator_ids, self.q_goal_left_hand)
        apply(self.torso_actuator_ids, self.q_goal_torso)

        # Hold others
        for aid, val in self.q_initial_other.items():
            self.data.ctrl[aid] = val

    def _apply_kinematic_control(self):
        def apply(addrs, goals):
            for i, addr in enumerate(addrs):
                if i < len(goals):
                    self.data.qpos[addr] = goals[i]

        apply(self.right_arm_qpos_addrs, self.q_goal_right)
        apply(self.left_arm_qpos_addrs, self.q_goal_left)
        apply(self.right_leg_qpos_addrs, self.q_goal_right_leg)
        apply(self.left_leg_qpos_addrs, self.q_goal_left_leg)
        apply(self.right_hand_qpos_addrs, self.q_goal_right_hand)
        apply(self.left_hand_qpos_addrs, self.q_goal_left_hand)
        apply(self.torso_qpos_addrs, self.q_goal_torso)

    # --- Mocap Body Management ---
    def setup_mocap_body(self, mocap_body_name="pelvis_mocap_mover"):
        self.mocap_body_name = mocap_body_name
        mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, mocap_body_name)
        if mocap_body_id >= 0:
            for i in range(self.model.nmocap):
                if self.model.body_mocapid[mocap_body_id] == i:
                    self.mocap_index = i
                    self.mocap_enabled = True
                    break
            if not self.mocap_enabled and self.model.nmocap > 0:
                self.mocap_index = 0
                self.mocap_enabled = True

        if self.debug:
            print(f"Mocap body setup: {self.mocap_enabled}, index: {self.mocap_index}")

    def update_mocap_body(self, position, R_world_body):
        if not self.mocap_enabled or self.mocap_index is None:
            return
        try:
            self.data.mocap_pos[self.mocap_index] = position

            quat_xyzw = Rotation.from_matrix(R_world_body).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

            # Filter
            if self.quat_prev_mocap is None:
                self.quat_prev_mocap = quat_wxyz
                quat_filtered = quat_wxyz
            else:
                if np.dot(self.quat_prev_mocap, quat_wxyz) < 0:
                    quat_wxyz = -quat_wxyz
                quat_filtered = ((1.0 - self.quat_filter_alpha) * self.quat_prev_mocap
                                 + self.quat_filter_alpha * quat_wxyz)
                quat_filtered /= np.linalg.norm(quat_filtered)
                self.quat_prev_mocap = quat_filtered.copy()

            self.data.mocap_quat[self.mocap_index] = quat_filtered
        except Exception as e:
            if self.debug:
                print(f"Error updating mocap: {e}")

    # --- Camera Management ---
    def setup_follow_camera(self, distance=2.0, height_offset=0.3, azimuth_offset=0.0,
                            elevation=-10.0):
        self.camera_distance = distance
        self.camera_height_offset = height_offset
        self.camera_azimuth_offset = azimuth_offset
        self.camera_elevation = elevation
        self.follow_camera_enabled = True

    def update_follow_camera(self, viewer):
        if not self.follow_camera_enabled or not self.mocap_enabled:
            return
        try:
            mocap_pos = self.data.mocap_pos[self.mocap_index].copy()
            mocap_pos[2] = 0.79  # Fixed height

            R_world_body = np.eye(3)  # Assume flat world alignment for camera frame
            robot_front = R_world_body @ np.array([1, 0, 0])
            robot_right = R_world_body @ np.array([0, -1, 0])

            az_rad = np.radians(self.camera_azimuth_offset)
            cam_back = -(np.cos(az_rad) * robot_front + np.sin(az_rad) * robot_right)

            cam_pos = mocap_pos + self.camera_distance * cam_back + np.array(
                [0, 0, self.camera_height_offset])
            lookat_pos = mocap_pos

            viewer.cam.lookat[:] = lookat_pos

            # Calculate azimuth/distance for viewer params
            diff = lookat_pos - cam_pos
            dist = np.linalg.norm(diff)
            azimuth = np.degrees(np.arctan2(diff[1], diff[0]))

            viewer.cam.distance = dist
            viewer.cam.azimuth = azimuth
            viewer.cam.elevation = self.camera_elevation
        except Exception as e:
            if self.debug:
                print(f"Camera update error: {e}")

    def get_current_joint_angles(self):
        self._update_current_positions()
        return {
            "right_arm": self.q_current_right,
            "left_arm": self.q_current_left,
            "right_leg": self.q_current_right_leg,
            "left_leg": self.q_current_left_leg,
            "right_hand": self.q_current_right_hand,
            "left_hand": self.q_current_left_hand,
            "torso": self.q_current_torso
        }

    def get_sew_transform(self):
        """Return a function that transforms points from the SEW frame to world."""
        # Find waist_pitch_joint ID
        waist_pitch_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch_joint")

        if waist_pitch_joint_id != -1:
            # Get the body ID driven by this joint
            body_id = self.model.jnt_bodyid[waist_pitch_joint_id]

            # Get joint position (anchor) in world frame
            pos_joint_world = self.data.xanchor[waist_pitch_joint_id]

            # Get body orientation in world frame
            rot_body_world = self.data.xmat[body_id].reshape(3, 3)

            # Offset from joint to SEW origin (in joint/body frame)
            offset = np.array([0, 0, 0.0])

            # SEW Origin in World Frame
            sew_origin_world = pos_joint_world + rot_body_world @ offset
            sew_rot_world = rot_body_world
        else:
            # Fallback
            sew_origin_world = np.zeros(3)
            sew_rot_world = np.eye(3)

        def to_world(p_local):
            return sew_origin_world + sew_rot_world @ p_local

        return to_world
