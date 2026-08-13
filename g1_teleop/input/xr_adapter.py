# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Interim XR device adapter.

TEMPORARY until the ``xrt_device`` repo exists: this adapter imports
``XRRTCBodyPoseDevice`` from a local SEW-Geometric-Teleop monolith checkout
(via an explicit ``monolith_path`` argument or the ``GEO_TELEOP_MONOLITH``
environment variable) and converts the legacy action dict it produces into a
:class:`geo_kin_core.types.RetargetFrame`.

Legacy action-dict schema (see the monolith CLAUDE.md, "Action dict" section;
produced by ``XRRTCBodyPoseDevice._default_process_bones_to_action``):

    R_world_upper_body (3,3)   human upper-body (shoulder-center) rotation
    p_world_upper_body (3,)    shoulder center position
    left_sew/right_sew (18,)   [S(3), E(3), W(3), R_world_wrist.flatten()(9)]
    head_rotation      (3,3)   head rotation in the upper-body frame
    left/right_fingers dict    per-finger keypoint dicts (body-centric)
    left/right_finger_tip_centroid, left/right_finger_mcp_centroid  (3,)|None
    left/right_gripper_val     float (thumb-index distance)
    ankle_to_body, R_lower_upper, body_center, left/right_hka       lower body
    tags               dict    AprilTag detections

The device itself needs the monolith's dependencies
(``xr_robot_teleop_server``, ``robosuite``, ``aiortc``, ...) — the import is
deferred to :class:`XRDeviceAdapter` construction so this module stays
importable everywhere.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from geo_kin_core.types import RetargetFrame, SEWPose

MONOLITH_ENV_VAR = "GEO_TELEOP_MONOLITH"

# Keys copied verbatim into RetargetFrame.extras (device/base-alignment data
# with no typed field yet).
_EXTRA_KEYS = (
    "body_center",
    "ankle_to_body",
    "tags",
    "left_finger_tip_centroid",
    "right_finger_tip_centroid",
    "left_finger_mcp_centroid",
    "right_finger_mcp_centroid",
)


def _opt_array(value, shape=None):
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def _opt_sew(flat18) -> Optional[SEWPose]:
    if flat18 is None:
        return None
    return SEWPose.from_flat18(np.asarray(flat18, dtype=float).reshape(18))


def action_to_retarget_frame(action: dict) -> Optional[RetargetFrame]:
    """Convert a legacy XR action dict into a geo_kin_core RetargetFrame.

    Includes ``R_lower_upper`` so the 3-DOF waist (torso) solve works — the
    old upper-body hw demo stubbed it to identity; the new session actually
    consumes it. Returns None for an empty/None action (device not streaming).
    """
    if not action:
        return None

    extras = {k: action[k] for k in _EXTRA_KEYS if action.get(k) is not None}

    return RetargetFrame(
        left_sew=_opt_sew(action.get("left_sew")),
        right_sew=_opt_sew(action.get("right_sew")),
        R_world_upper_body=_opt_array(action.get("R_world_upper_body"), (3, 3)),
        p_world_upper_body=_opt_array(action.get("p_world_upper_body"), (3,)),
        head_rotation=_opt_array(action.get("head_rotation"), (3, 3)),
        R_lower_upper=_opt_array(action.get("R_lower_upper"), (3, 3)),
        left_fingers=action.get("left_fingers"),
        right_fingers=action.get("right_fingers"),
        left_hka=_opt_array(action.get("left_hka")),
        right_hka=_opt_array(action.get("right_hka")),
        left_gripper_val=action.get("left_gripper_val"),
        right_gripper_val=action.get("right_gripper_val"),
        extras=extras,
    )


def resolve_monolith_path(monolith_path=None) -> Path:
    """Resolve the SEW-Geometric-Teleop checkout path (arg > env var)."""
    path = monolith_path or os.environ.get(MONOLITH_ENV_VAR)
    if not path:
        raise RuntimeError(
            "XRDeviceAdapter needs a SEW-Geometric-Teleop checkout until the "
            "xrt_device repo exists. Pass monolith_path=... or set the "
            f"{MONOLITH_ENV_VAR} environment variable."
        )
    path = Path(path).expanduser().resolve()
    if not (path / "projects" / "shared_devices" / "xr_robot_teleop_client.py").exists():
        raise RuntimeError(
            f"{path} does not look like a SEW-Geometric-Teleop checkout "
            "(projects/shared_devices/xr_robot_teleop_client.py not found)"
        )
    return path


class XRDeviceAdapter:
    """XRRTCBodyPoseDevice (monolith) -> RetargetFrame adapter.

    TEMPORARY: imports the device class out of a monolith checkout. Once the
    xrt_device repo is split out, this class shrinks to a thin wrapper around
    that package (the RetargetFrame conversion stays).
    """

    def __init__(self, monolith_path=None, **device_kwargs):
        """
        Args:
            monolith_path: SEW-Geometric-Teleop checkout root; falls back to
                the GEO_TELEOP_MONOLITH environment variable.
            **device_kwargs: Forwarded to XRRTCBodyPoseDevice (record_data,
                output_dir, ndigits, process_bones_to_action_fn, ...).
        """
        root = resolve_monolith_path(monolith_path)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from projects.shared_devices.xr_robot_teleop_client import XRRTCBodyPoseDevice
        except ImportError as e:
            raise ImportError(
                "Failed to import XRRTCBodyPoseDevice from the monolith at "
                f"{root}. The XR device needs the monolith's dependencies "
                "(xr_robot_teleop_server, robosuite, ...) installed in this "
                "environment."
            ) from e

        device_kwargs.setdefault("env", None)
        self.device = XRRTCBodyPoseDevice(**device_kwargs)

    @property
    def is_connected(self) -> bool:
        return bool(self.device.is_connected)

    def get_raw_action(self) -> Optional[dict]:
        """Latest legacy action dict from the device (None if not streaming)."""
        return self.device.get_controller_state()

    def get_frame(self) -> Optional[RetargetFrame]:
        """Latest input converted to a RetargetFrame (None if not streaming)."""
        return action_to_retarget_frame(self.get_raw_action())

    def cleanup(self):
        """Flush CSV recording buffers (when record_data=True)."""
        cleanup = getattr(self.device, "cleanup_recording", None)
        if cleanup is not None:
            cleanup()
