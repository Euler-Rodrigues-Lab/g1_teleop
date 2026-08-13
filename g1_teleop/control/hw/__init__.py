# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Unitree G1 hardware layer (DDS body control + Inspire hand Modbus).

``unitree_sdk2py`` is imported lazily inside :mod:`.unitree_api_utils` and
``pymodbus`` lazily inside :mod:`.inspire_hand`, so this subpackage is
importable on sim-only installs; constructing a :class:`Controller` /
:class:`InspireHandController` (or calling :func:`init_channel_factory`) is
what requires the hardware SDKs (``pip install g1-teleop[hw]``).
"""

from .config import DEFAULT_CONFIG_PATH, Config
from .inspire_hand import InspireHandController
from .unitree_api_utils import Controller, init_channel_factory

__all__ = ["Config", "DEFAULT_CONFIG_PATH", "Controller", "InspireHandController",
           "init_channel_factory"]
