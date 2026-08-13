# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Input devices (interim adapters until the device repos are split out)."""

from .xr_adapter import XRDeviceAdapter, action_to_retarget_frame

__all__ = ["XRDeviceAdapter", "action_to_retarget_frame"]
