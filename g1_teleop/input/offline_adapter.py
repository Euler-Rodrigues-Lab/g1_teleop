# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Offline CSV playback adapter: recorded OpenXR body pose -> RetargetFrame.

TEMPORARY, like :mod:`g1_teleop.input.xr_adapter`: the CSV reader
(``CSVDataReader``) and the bones->action conversion live in a
SEW-Geometric-Teleop checkout and pull device-side dependencies (pandas,
``xr_robot_teleop_server`` bone schemas). Point at a checkout with
``monolith_path=`` or the ``GEO_TELEOP_MONOLITH`` environment variable; once
the ``xrt_device`` repo is split out this class shrinks to a thin wrapper
around that package and the RetargetFrame conversion stays.

The recordings themselves are not vendored here — pass ``--csv_file`` (the
monolith ships them under ``References/recordings/``).
"""

import sys
from typing import Optional, Tuple

from geo_kin_core.types import RetargetFrame

from .xr_adapter import action_to_retarget_frame, resolve_monolith_path


class OfflineCSVAdapter:
    """Recorded-CSV playback source with the same output type as the XR device.

    Mirrors :class:`g1_teleop.input.XRDeviceAdapter` so demos can swap a live
    headset for a recording without touching the solve/control path.
    """

    def __init__(self, csv_file, playback_speed: float = 1.0, loop: bool = True,
                 monolith_path=None, **reader_kwargs):
        """
        Args:
            csv_file: Recorded OpenXR body-pose CSV.
            playback_speed: Time-scale multiplier (1.0 = real time).
            loop: Restart playback when the recording ends.
            monolith_path: SEW-Geometric-Teleop checkout root; falls back to
                the GEO_TELEOP_MONOLITH environment variable.
            **reader_kwargs: Forwarded to CSVDataReader.
        """
        root = resolve_monolith_path(monolith_path)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from projects.shared_devices.offline_openxr_pose_reader import CSVDataReader
            from projects.shared_devices.xr_robot_teleop_client import XRRTCBodyPoseDevice
        except ImportError as e:
            raise ImportError(
                f"Failed to import the offline CSV reader from the monolith at {root}. "
                "It needs the monolith's device dependencies (pandas, "
                "xr_robot_teleop_server, ...) installed in this environment."
            ) from e

        self._bones_to_action = XRRTCBodyPoseDevice._default_process_bones_to_action
        self.reader = CSVDataReader(str(csv_file), playback_speed, loop=loop, **reader_kwargs)
        self.loop = loop

    @property
    def duration(self) -> float:
        """Recording length in seconds (before playback-speed scaling)."""
        return float(self.reader.get_duration())

    def get_bones_at_time(self, elapsed_time: float):
        """Raw bone list at `elapsed_time` (None past the end when loop=False)."""
        try:
            return self.reader.get_bones_at_time(elapsed_time)
        except Exception:
            return None

    def get_frame_at_time(
        self, elapsed_time: float
    ) -> Tuple[Optional[RetargetFrame], Optional[object]]:
        """Return ``(frame, bones)`` at `elapsed_time`.

        ``bones`` is passed through so demos can drive the human-skeleton
        overlay; both are None once a non-looping recording is exhausted.
        """
        bones = self.get_bones_at_time(elapsed_time)
        if bones is None:
            return None, None
        return action_to_retarget_frame(self._bones_to_action(bones)), bones
