# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Hardware configuration loader (g1.yaml).

Standalone implementation extracted from the original controller.
"""

from pathlib import Path

import numpy as np
import yaml

# Default config shipped with the package.
DEFAULT_CONFIG_PATH = Path(__file__).parent / "g1.yaml"


class Config:
    def __init__(self, file_path=DEFAULT_CONFIG_PATH) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]
            self.controller_topic = config["controller_topic"]

            self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            self.arm_waist_joint2motor_idx = config["arm_waist_joint2motor_idx"]
            self.arm_waist_kps = config["arm_waist_kps"]
            self.arm_waist_kds = config["arm_waist_kds"]
            self.arm_waist_target = np.array(config["arm_waist_target"], dtype=np.float32)

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
