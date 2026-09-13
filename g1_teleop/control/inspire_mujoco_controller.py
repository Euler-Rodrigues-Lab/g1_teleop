# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""MuJoCo controller for the G1 with Inspire hands mounted.

Standalone implementation extracted from the original controller.
Subclasses :class:`G1FullBodyMuJoCoController`, swapping the psyonic finger
joints for the 6 active Inspire finger joints per hand and enforcing the
model's joint equalities (passive distal finger joints) when writing qpos
directly in kinematic mode.

``q_goal_right_hand`` / ``q_goal_left_hand`` are wired to the Inspire hand
actuators in the packing order the inspire retargeting session emits:
``[little, ring, middle, index, thumb_2 (flexion), thumb_1 (rotation)]``.

``set_joint_goals`` is inherited unchanged from the base controller, so it
stays duck-typed exactly the same way: a goals dict with ``q_goal_*`` keys OR
any attribute-style object exposing the same names (e.g.
``geo_kin_core.types.RetargetOutput``); ``None`` means "keep previous goal".
"""

import mujoco
import numpy as np

from .mujoco_controller import G1FullBodyMuJoCoController


class G1InspireHandMuJoCoController(G1FullBodyMuJoCoController):
    """G1 + Inspire hand controller: joint position goals + passive-joint equalities."""

    def __init__(self, mujoco_model, mujoco_data, debug=False, timing_enabled=False,
                 singularity_overlay_enabled=False):
        super().__init__(mujoco_model, mujoco_data, debug=debug,
                         timing_enabled=timing_enabled,
                         singularity_overlay_enabled=singularity_overlay_enabled)
        self._setup_joint_equality_constraints()

        if self.debug:
            print("G1 Inspire Hand MuJoCo Controller initialized")

    def _setup_joint_indices(self):
        """Setup joint indices and names for all body parts (Inspire hand variant)."""
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
        # Active Inspire finger joints, in q_goal_*_hand packing order.
        self.right_hand_joint_names = [
            "right_little_1_joint", "right_ring_1_joint", "right_middle_1_joint",
            "right_index_1_joint", "right_thumb_2_joint", "right_thumb_1_joint"
        ]
        self.left_hand_joint_names = [
            "left_little_1_joint", "left_ring_1_joint", "left_middle_1_joint",
            "left_index_1_joint", "left_thumb_2_joint", "left_thumb_1_joint"
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

        # Actuator indices, in joint-name order (the goals packing order),
        # not model order — this is what keeps q_goal_*_hand[i] wired to the
        # matching Inspire actuator.
        actuator_name2id = {}
        for i in range(self.model.nu):
            name = self.model.actuator(i).name
            if name:
                actuator_name2id[name] = i

        def get_actuator_ids(names):
            return [actuator_name2id[name] for name in names if name in actuator_name2id]

        self.right_arm_actuator_ids = get_actuator_ids(self.right_arm_joint_names)
        self.left_arm_actuator_ids = get_actuator_ids(self.left_arm_joint_names)
        self.right_leg_actuator_ids = get_actuator_ids(self.right_leg_joint_names)
        self.left_leg_actuator_ids = get_actuator_ids(self.left_leg_joint_names)
        self.right_hand_actuator_ids = get_actuator_ids(self.right_hand_joint_names)
        self.left_hand_actuator_ids = get_actuator_ids(self.left_hand_joint_names)
        self.torso_actuator_ids = get_actuator_ids(self.torso_joint_names)

        controlled_names = set(self.right_arm_joint_names + self.left_arm_joint_names +
                               self.right_leg_joint_names + self.left_leg_joint_names +
                               self.right_hand_joint_names + self.left_hand_joint_names +
                               self.torso_joint_names)
        self.other_actuator_ids = [
            i for i in range(self.model.nu)
            if self.model.actuator(i).name not in controlled_names
        ]

    def _store_initial_positions(self):
        """Store initial positions of other (non-teleoperated) actuators.

        Reads the driven joint's qpos through the actuator transmission where
        possible (the monolith behavior), so nonzero initial poses are held.
        """
        self.q_initial_other = {}
        for aid in self.other_actuator_ids:
            self.q_initial_other[aid] = 0.0
            try:
                jnt_id = self.model.actuator_trnid[aid, 0]
                qpos_addr = self.model.jnt_qposadr[jnt_id]
                self.q_initial_other[aid] = self.data.qpos[qpos_addr]
            except (IndexError, ValueError):
                pass

    # --- Joint equality constraints (passive Inspire finger joints) ---
    def _setup_joint_equality_constraints(self):
        """Cache joint-equality constraints for kinematic-mode updates."""
        self.joint_equality_constraints = []
        if self.model.neq <= 0:
            return

        for eq_id in range(self.model.neq):
            if self.model.eq_type[eq_id] != mujoco.mjtEq.mjEQ_JOINT:
                continue

            joint1_id = int(self.model.eq_obj1id[eq_id])
            joint2_id = int(self.model.eq_obj2id[eq_id])
            if joint1_id < 0:
                continue

            qpos1_addr = int(self.model.jnt_qposadr[joint1_id])
            qpos2_addr = int(self.model.jnt_qposadr[joint2_id]) if joint2_id >= 0 else None
            polycoef = np.array(self.model.eq_data[eq_id, :5], dtype=float)

            self.joint_equality_constraints.append({
                "eq_id": eq_id,
                "qpos1_addr": qpos1_addr,
                "qpos2_addr": qpos2_addr,
                "polycoef": polycoef,
            })

        if self.debug:
            print(f"Cached {len(self.joint_equality_constraints)} joint equality constraints")

    def _apply_joint_equalities_in_kinematic_mode(self):
        """Manually enforce joint equalities when writing qpos directly."""
        if not self.joint_equality_constraints:
            return

        for constraint in self.joint_equality_constraints:
            eq_id = constraint["eq_id"]
            if hasattr(self.data, "eq_active") and not self.data.eq_active[eq_id]:
                continue

            q2 = 0.0
            if constraint["qpos2_addr"] is not None:
                q2 = float(self.data.qpos[constraint["qpos2_addr"]])

            c0, c1, c2, c3, c4 = constraint["polycoef"]
            self.data.qpos[constraint["qpos1_addr"]] = (
                ((((c4 * q2) + c3) * q2 + c2) * q2 + c1) * q2 + c0
            )

    def _apply_kinematic_control(self):
        super()._apply_kinematic_control()
        # Direct qpos writes bypass MuJoCo's constraint solver, so enforce
        # joint equalities explicitly (e.g., passive Inspire hand joints).
        self._apply_joint_equalities_in_kinematic_mode()
