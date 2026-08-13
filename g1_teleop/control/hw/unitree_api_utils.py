# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Unitree G1 low-level DDS hardware interface.

Ported from the monolith
(SEW-Geometric-Teleop/projects/g1_upper_body_teleop/unitree_api_utils.py).
Changes vs the monolith:

* ``unitree_sdk2py`` imports are LAZY (inside ``_import_sdk``): sim-only
  installs can import this module (and everything above it) without the SDK.
  Install the SDK with ``pip install g1-teleop[hw]``.
* dropped the unused ``import torch``.
* added ``enter_damping`` (the monolith demo called it but never defined it)
  and the ``init_channel_factory`` wrapper so demos never import the SDK
  directly.

Upper-body motor ordering follows ``arm_waist_joint2motor_idx`` in g1.yaml:
``[torso(3), left_arm(7), right_arm(7)]``.
"""

import time

import numpy as np

from .config import Config
from .helpers import MotorMode, create_damping_cmd, init_cmd_go, init_cmd_hg


def _import_sdk():
    """Import unitree_sdk2py lazily; raise a helpful error when absent."""
    try:
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import (
            unitree_go_msg_dds__LowCmd_,
            unitree_go_msg_dds__LowState_,
            unitree_go_msg_dds__WirelessController_,
            unitree_hg_msg_dds__LowCmd_,
            unitree_hg_msg_dds__LowState_,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
        from unitree_sdk2py.utils.crc import CRC
    except ImportError as e:  # pragma: no cover - exercised only without SDK
        raise ImportError(
            "unitree_sdk2py is required for real-robot control. "
            "Install it with: pip install g1-teleop[hw]"
        ) from e
    return {
        "ChannelFactoryInitialize": ChannelFactoryInitialize,
        "ChannelPublisher": ChannelPublisher,
        "ChannelSubscriber": ChannelSubscriber,
        "LowCmdHG": LowCmdHG,
        "LowCmdGo": LowCmdGo,
        "LowStateHG": LowStateHG,
        "LowStateGo": LowStateGo,
        "WirelessController_": WirelessController_,
        "CRC": CRC,
        "new_lowcmd_hg": unitree_hg_msg_dds__LowCmd_,
        "new_lowstate_hg": unitree_hg_msg_dds__LowState_,
        "new_lowcmd_go": unitree_go_msg_dds__LowCmd_,
        "new_lowstate_go": unitree_go_msg_dds__LowState_,
        "new_wireless": unitree_go_msg_dds__WirelessController_,
    }


def init_channel_factory(domain_id: int = 0, net_interface: str = None):
    """Initialize the Unitree DDS channel factory (call once before Controller)."""
    sdk = _import_sdk()
    if net_interface is None:
        sdk["ChannelFactoryInitialize"](domain_id)
    else:
        sdk["ChannelFactoryInitialize"](domain_id, net_interface)


class Controller:
    """Low-level DDS controller for the G1 upper body (arms + waist)."""

    def __init__(self, config: Config) -> None:
        self.config = config
        sdk = self._sdk = _import_sdk()
        self._crc = sdk["CRC"]()

        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = sdk["new_lowcmd_hg"]()
            self.low_state = sdk["new_lowstate_hg"]()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = sdk["ChannelPublisher"](config.lowcmd_topic, sdk["LowCmdHG"])
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = sdk["ChannelSubscriber"](
                config.lowstate_topic, sdk["LowStateHG"])
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = sdk["new_lowcmd_go"]()
            self.low_state = sdk["new_lowstate_go"]()

            self.lowcmd_publisher_ = sdk["ChannelPublisher"](config.lowcmd_topic, sdk["LowCmdGo"])
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = sdk["ChannelSubscriber"](
                config.lowstate_topic, sdk["LowStateGo"])
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        self.wireless_controller = sdk["new_wireless"]()
        self.controller_subscriber = sdk["ChannelSubscriber"](
            config.controller_topic, sdk["WirelessController_"])

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        # Cached command vectors for fast control updates.
        self._dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        self._dof_kps = self.config.kps + self.config.arm_waist_kps
        self._dof_kds = self.config.kds + self.config.arm_waist_kds
        self._dof_default_pos = np.concatenate(
            (self.config.default_angles, self.config.arm_waist_target), axis=0
        )

        if not (
            len(self._dof_idx)
            == len(self._dof_kps)
            == len(self._dof_kds)
            == len(self._dof_default_pos)
        ):
            raise ValueError("Invalid config: inconsistent DOF index/gain/default lengths")

        self._motor_to_default = {
            motor_idx: float(q) for motor_idx, q in zip(self._dof_idx, self._dof_default_pos)
        }
        self._motor_to_kp = {
            motor_idx: float(kp) for motor_idx, kp in zip(self._dof_idx, self._dof_kps)
        }
        self._motor_to_kd = {
            motor_idx: float(kd) for motor_idx, kd in zip(self._dof_idx, self._dof_kds)
        }

        self._upper_body_idx = self.config.arm_waist_joint2motor_idx
        if len(self._upper_body_idx) < 17:
            raise ValueError(
                f"Expected at least 17 upper-body motor indices, got {len(self._upper_body_idx)}"
            )
        self._torso_idx = self._upper_body_idx[0:3]
        self._left_idx = self._upper_body_idx[3:10]
        self._right_idx = self._upper_body_idx[10:17]

    def LowStateHgHandler(self, msg):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine

    def LowStateGoHandler(self, msg):
        self.low_state = msg

    def send_cmd(self, cmd):
        cmd.crc = self._crc.Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")

    def move_to_default_pos(self):
        """Interpolate all configured DOFs to their default pose over ~2 s."""
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        dof_size = len(self._dof_idx)

        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[self._dof_idx[i]].q

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = self._dof_idx[j]
                target_pos = self._dof_default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = (
                    init_dof_pos[j] * (1 - alpha) + target_pos * alpha)
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self._dof_kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = self._dof_kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def get_current_upper_body(self):
        """Return current upper-body joint angles as (torso, left_arm, right_arm)."""
        q_torso = np.array(
            [self.low_state.motor_state[i].q for i in self._torso_idx], dtype=np.float64)
        q_left = np.array(
            [self.low_state.motor_state[i].q for i in self._left_idx], dtype=np.float64)
        q_right = np.array(
            [self.low_state.motor_state[i].q for i in self._right_idx], dtype=np.float64)
        return q_torso, q_left, q_right

    def send_upper_body_targets(
        self,
        q_goal_torso,
        q_goal_left,
        q_goal_right,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
    ):
        """
        Send torso/left/right upper-body targets while keeping non-commanded joints at defaults.
        Upper-body ordering follows arm_waist_joint2motor_idx in g1.yaml:
        [torso(3), left_arm(7), right_arm(7)].
        """
        q_goal_torso = np.asarray(q_goal_torso, dtype=np.float64).reshape(-1)
        q_goal_left = np.asarray(q_goal_left, dtype=np.float64).reshape(-1)
        q_goal_right = np.asarray(q_goal_right, dtype=np.float64).reshape(-1)
        if q_goal_torso.size != 3 or q_goal_left.size != 7 or q_goal_right.size != 7:
            raise ValueError(
                f"Expected torso/left/right sizes (3,7,7), got "
                f"({q_goal_torso.size},{q_goal_left.size},{q_goal_right.size})"
            )

        upper_targets = np.concatenate((q_goal_torso, q_goal_left, q_goal_right), axis=0)
        targets_by_motor_idx = {
            motor_idx: float(q) for motor_idx, q in zip(self._upper_body_idx, upper_targets)
        }

        for motor_idx in self._dof_idx:
            target_q = targets_by_motor_idx.get(motor_idx, self._motor_to_default[motor_idx])
            self.low_cmd.motor_cmd[motor_idx].mode = 1  # 1: enable
            self.low_cmd.motor_cmd[motor_idx].q = float(target_q)
            self.low_cmd.motor_cmd[motor_idx].qd = 0.0
            self.low_cmd.motor_cmd[motor_idx].kp = self._motor_to_kp[motor_idx] * kp_scale
            self.low_cmd.motor_cmd[motor_idx].kd = self._motor_to_kd[motor_idx] * kd_scale
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0

        self.send_cmd(self.low_cmd)

    def enter_damping(self, duration_s: float = 0.3):
        """Send a pure-damping command for `duration_s` (safe shutdown).

        The monolith demo called this at exit but never defined it; implemented
        here with the ported create_damping_cmd helper.
        """
        create_damping_cmd(self.low_cmd)
        end = time.time() + max(float(duration_s), 0.0)
        while True:
            self.send_cmd(self.low_cmd)
            if time.time() >= end:
                break
            time.sleep(self.config.control_dt)
