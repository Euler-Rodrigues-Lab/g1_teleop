# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Inspire RH56DFTP dexterous hand hardware interface (Modbus TCP).

Based on the Inspire hand interface by Roman Mineyev. Changes vs the monolith:

* ``pymodbus`` is imported LAZILY (inside ``_import_modbus``): sim-only
  installs can import this module without the hardware SDK. Install it with
  ``pip install g1-teleop[hw]``.
* dropped the unused ``inspire_sdkpy`` DDS imports and the matplotlib-based
  sensor-visualization imports (the Modbus-direct rewrite never used them);
  the standalone constant-step CLI test was left in the monolith.
* ``set_joint_goals`` is duck-typed like the MuJoCo controllers: it accepts a
  goals dict with ``q_goal_right_hand`` / ``q_goal_left_hand`` keys OR any
  attribute-style object exposing the same names (e.g.
  ``geo_kin_core.types.RetargetOutput``); ``None`` = keep previous goal.

Finger goal packing order (radians, from the inspire retargeting session):
``[little, ring, middle, index, thumb_2 (flexion), thumb_1 (rotation)]``.
"""

from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Mapping
from typing import Dict

import numpy as np

_SIZEOF_C_SHORT: int = ctypes.sizeof(ctypes.c_short)  # in Bytes

# Factory-default Modbus TCP endpoints of the hands on the G1 payload network.
DEFAULT_LEFT_IP = "192.168.123.210"
DEFAULT_RIGHT_IP = "192.168.123.211"
DEFAULT_PORT = 6000


def _import_modbus():
    """Import pymodbus lazily; raise a helpful error when absent."""
    try:
        from pymodbus.client import ModbusTcpClient
    except ImportError as e:  # pragma: no cover - exercised only without SDK
        raise ImportError(
            "pymodbus is required for Inspire hand hardware control. "
            "Install it with: pip install g1-teleop[hw]"
        ) from e
    return ModbusTcpClient


def _get_goal(goals, key):
    """Read a joint-group goal from a dict or an attribute-style object."""
    if isinstance(goals, Mapping):
        return goals.get(key)
    return getattr(goals, key, None)


class InspireHandActStates:
    """Inspire hand actuation mode flags (bit-OR combinable).

    Reference: INSPIRE-ROBOTS RH56DFTP User Manual V1.0.0.
    """

    IDLE: int = 0b0000
    ANGLE: int = 0b0001
    POSITION: int = 0b0010
    FORCE: int = 0b0100
    VELOCITY: int = 0b1000


class InspireHandRegAddr:
    """Inspire hand Modbus register addresses.

    Reference: INSPIRE-ROBOTS RH56DFTP User Manual V1.0.0.
    """

    # W/R Registers
    ## 1 Byte Config
    HAND_ID: int = 1000
    REDU_RATIO: int = 1002
    CLEAR_ERROR: int = 1004
    SAVE: int = 1005
    RESET_PARA: int = 1006
    GESTURE_FORCE_CLB: int = 1009
    ## 12 Byte Config
    DEFAULT_SPEED_SET: int = 1032
    DEFAULT_FORCE_SET: int = 1044
    POS_SET: int = 1474
    ANGLE_SET: int = 1486
    FORCE_SET: int = 1498
    SPEED_SET: int = 1522

    # R-Only Joint Sensors
    ## 12 Byte Config
    POS_ACT: int = 1534
    ANGLE_ACT: int = 1546
    FORCE_ACT: int = 1582
    CURRENT: int = 1594
    ## 6 Byte Config
    ERROR: int = 1606
    STATUS: int = 1612
    TEMP: int = 1618

    # IP setting is W/R
    ## All 1 Byte Config
    IP_PART1: int = 1700
    IP_PART2: int = 1701
    IP_PART3: int = 1702
    IP_PART4: int = 1703

    # R-Only Finger Sensors
    ## All 370 Byte Config
    ### 18 (3, 3) from tip, 192 (12, 8) from top, 160 (10, 8) from pulp
    FINGERONE_TOUCH: int = 3000  # Pinky
    FINGERTWO_TOUCH: int = 3370  # Ring
    FINGERTHE_TOUCH: int = 3740  # Middle
    FINGERFOR_TOUCH: int = 4110  # Index
    ### 18 (3, 3) from tip, 192 (12, 8) from top, 18 (3, 3) from middle, 192 (12, 8) from pulp
    FINGERFIV_TOUCH: int = 4480  # Thumb
    ### 224 (14, 8) from palm
    FINGERPALM_TOUCH: int = 4900  # Palm


class InspireHandController:
    """High-level controller for Inspire robotic hands over direct Modbus TCP.

    A background worker thread streams the current position goals to the
    hand(s) at ~50 Hz whenever engaged, and synchronously refreshes the joint
    state / touch sensor snapshots readable via :meth:`read`.

    Example:
        >>> hand = InspireHandController('b')  # both hands
        >>> hand.set_joint_goals({'q_goal_right_hand': np.zeros(6)})
        >>> hand.apply_control(engaged=True)
    """

    def __init__(self, hand_side='b', network=None, initialize_dds=False,
                 left_ip=DEFAULT_LEFT_IP, right_ip=DEFAULT_RIGHT_IP,
                 port=DEFAULT_PORT,
                 RegAddr=InspireHandRegAddr, ActStates=InspireHandActStates):
        """
        Initialize controller for left, right, or both hands.

        Args:
            hand_side (str): 'l' left, 'r' right, or 'b' both. Defaults to 'b'.
            network (str, optional): Unused (kept for monolith call-site
                compatibility; the Modbus-direct rewrite bypasses DDS).
            initialize_dds (bool): Unused (kept for monolith call-site
                compatibility).
            left_ip / right_ip (str): Modbus TCP addresses of the hands.
            port (int): Modbus TCP port.
        """
        del network, initialize_dds  # DDS path removed in the Modbus rewrite

        # Register addresses and actuation mode flags
        self.RegAddr = RegAddr
        self.ActStates = ActStates
        self.states: dict[str, dict[str, list]] = {
            hand: {
                'POS_ACT': [0] * 6,
                'ANGLE_ACT': [0] * 6,
                'FORCE_ACT': [0] * 6,
                'CURRENT': [0] * 6,
                'ERROR': [0] * 6,
                'STATUS': [0] * 6,
                'TEMP': [0] * 6,
            }
            for hand in ('Left', 'Right')
        }
        self.touch: dict[str, dict[str, np.ndarray]] = {
            hand: {
                'fingerone': np.zeros((18 + 192 + 160) // _SIZEOF_C_SHORT, dtype=int),
                'fingertwo': np.zeros((18 + 192 + 160) // _SIZEOF_C_SHORT, dtype=int),
                'fingerthree': np.zeros((18 + 192 + 160) // _SIZEOF_C_SHORT, dtype=int),
                'fingerfour': np.zeros((18 + 192 + 160) // _SIZEOF_C_SHORT, dtype=int),
                'fingerfive': np.zeros((18 + 192 + 18 + 192) // _SIZEOF_C_SHORT, dtype=int),
                'fingerpalm': np.zeros(224 // _SIZEOF_C_SHORT, dtype=int),
            }
            for hand in ('Left', 'Right')
        }

        # Protect shared state/touch dictionaries from concurrent read/write races.
        self._data_lock = threading.Lock()

        if hand_side not in ['l', 'r', 'b']:
            raise ValueError("hand_side must be 'l' (left), 'r' (right), or 'b' (both)")
        self.hand_side = hand_side

        ModbusTcpClient = _import_modbus()
        print("Creating Modbus TCP connections directly...")
        self.device_id = 1

        if self.hand_side == 'b':
            self.client_left = ModbusTcpClient(left_ip, port=port)
            self.client_left.connect()
            self.client_right = ModbusTcpClient(right_ip, port=port)
            self.client_right.connect()
            self.client = None

            # Serialize Modbus I/O per client across actuator and sensor threads.
            self._io_lock_left = threading.Lock()
            self._io_lock_right = threading.Lock()
        else:
            ip = left_ip if self.hand_side == 'l' else right_ip
            self.client = ModbusTcpClient(ip, port=port)
            self.client.connect()

            # Single client in left-only or right-only mode.
            self._io_lock = threading.Lock()

        print("Modbus clients initialized")

        # MuJoCo joint ranges (min, max) in radians.
        # Order in goals array: [little, ring, middle, index, thumb_2, thumb_1]
        self.joint_ranges = {
            'pinky': [0.0, 1.4381],
            'ring': [0.0, 1.4381],
            'middle': [0.0, 1.4381],
            'index': [0.0, 1.4381],
            'thumb_1': [0.0, 1.1641],
            'thumb_2': [0.0, 0.5864],
        }
        # Max values in goals order (mins are all 0).
        self.range_list = [
            self.joint_ranges['pinky'][1],
            self.joint_ranges['ring'][1],
            self.joint_ranges['middle'][1],
            self.joint_ranges['index'][1],
            self.joint_ranges['thumb_2'][1],  # thumb_2 is bend
            self.joint_ranges['thumb_1'][1],  # thumb_1 is rotate
        ]

        # Initialize default goals to avoid startup races before first external goal set.
        self.q_goal_left_hand = [0] * 6
        self.q_goal_right_hand = [0] * 6

        # Start Modbus worker thread to avoid blocking the main loop.
        self._is_engaged = False
        self._running = True
        self._sync_sensors_with_actuation = True
        self._actuator_thread = threading.Thread(target=self._actuator_worker, daemon=True)
        self._actuator_thread.start()

    # --- Register decoding helpers ---
    @staticmethod
    def _decode_int16_registers(registers: list[int]) -> list[int]:
        """Decode unsigned 16-bit register words to signed int16 values."""
        decoded: list[int] = []
        for reg in registers:
            reg16 = reg & 0xFFFF
            decoded.append(reg16 - 0x10000 if reg16 & 0x8000 else reg16)
        return decoded

    @staticmethod
    def _unpack_registers_to_bytes(registers: list[int]) -> list[int]:
        """Split each 16-bit register into [high_byte, low_byte] uint8 values."""
        byte_list: list[int] = []
        for reg in registers:
            reg16 = reg & 0xFFFF
            byte_list.append((reg16 >> 8) & 0xFF)
            byte_list.append(reg16 & 0xFF)
        return byte_list

    # --- Modbus I/O ---
    def _get_io_lock(self, client):
        """Return the lock guarding the given Modbus client connection."""
        if self.hand_side == 'b':
            if client is self.client_left:
                return self._io_lock_left
            if client is self.client_right:
                return self._io_lock_right
            raise ValueError("unknown Modbus client for lock lookup")
        return self._io_lock

    def _read_input_registers(self, client, address: int, count: int) -> list[int] | None:
        """Read input registers and return None on Modbus exception responses."""
        with self._get_io_lock(client):
            response = client.read_holding_registers(address, count, slave=self.device_id)
        if response.isError():
            print(f"Modbus read error at address={address}, count={count}: {response}")
            return None
        return response.registers

    def _read_touch_field(self, client, segments: list[tuple[int, int]]) -> np.ndarray | None:
        """Read a touch field from multiple register segments and concatenate."""
        values: list[int] = []
        for addr, count in segments:
            registers = self._read_input_registers(client, addr, count)
            if registers is None:
                return None
            values.extend(registers)
        return np.array(values, dtype=int)

    @staticmethod
    def _get_touch_segments(RegAddr) -> dict[str, list[tuple[int, int]]]:
        """Return touch register segments for a full hand snapshot."""
        return {
            'fingerone': [
                (RegAddr.FINGERONE_TOUCH, 18 // _SIZEOF_C_SHORT),
                (RegAddr.FINGERONE_TOUCH + 18, 192 // _SIZEOF_C_SHORT),
                (RegAddr.FINGERONE_TOUCH + 18 + 192, 160 // _SIZEOF_C_SHORT),
            ],
            'fingertwo': [
                (RegAddr.FINGERTWO_TOUCH, 18 // _SIZEOF_C_SHORT),
                (RegAddr.FINGERTWO_TOUCH + 18, 192 // _SIZEOF_C_SHORT),
                (RegAddr.FINGERTWO_TOUCH + 18 + 192, 160 // _SIZEOF_C_SHORT),
            ],
            'fingerthree': [
                (RegAddr.FINGERTHE_TOUCH, 18 // _SIZEOF_C_SHORT),
                (RegAddr.FINGERTHE_TOUCH + 18, 192 // _SIZEOF_C_SHORT),
                (RegAddr.FINGERTHE_TOUCH + 18 + 192, 160 // _SIZEOF_C_SHORT),
            ],
            'fingerfour': [
                (RegAddr.FINGERFOR_TOUCH, 18 // _SIZEOF_C_SHORT),
                (RegAddr.FINGERFOR_TOUCH + 18, 192 // _SIZEOF_C_SHORT),
                (RegAddr.FINGERFOR_TOUCH + 18 + 192, 160 // _SIZEOF_C_SHORT),
            ],
            'fingerfive': [
                (RegAddr.FINGERFIV_TOUCH, 18 // _SIZEOF_C_SHORT),
                (RegAddr.FINGERFIV_TOUCH + 18, 192 // _SIZEOF_C_SHORT),
                (RegAddr.FINGERFIV_TOUCH + 18 + 192, 18 // _SIZEOF_C_SHORT),
                (RegAddr.FINGERFIV_TOUCH + 18 + 192 + 18, 192 // _SIZEOF_C_SHORT),
            ],
            'fingerpalm': [
                (RegAddr.FINGERPALM_TOUCH, 224 // _SIZEOF_C_SHORT),
            ],
        }

    def _read_joint_states_once(self, client) -> dict[str, list[int]]:
        """Read all joint-related state registers for one hand."""
        state_updates: dict[str, list[int]] = {}

        for state_key, addr in (
            ('POS_ACT', self.RegAddr.POS_ACT),
            ('ANGLE_ACT', self.RegAddr.ANGLE_ACT),
            ('FORCE_ACT', self.RegAddr.FORCE_ACT),
            ('CURRENT', self.RegAddr.CURRENT),
        ):
            registers = self._read_input_registers(client, addr, 6)
            if registers is not None:
                state_updates[state_key] = self._decode_int16_registers(registers)

        for state_key, addr in (
            ('ERROR', self.RegAddr.ERROR),
            ('STATUS', self.RegAddr.STATUS),
            ('TEMP', self.RegAddr.TEMP),
        ):
            registers = self._read_input_registers(client, addr, 3)
            if registers is not None:
                state_updates[state_key] = self._unpack_registers_to_bytes(registers)

        return state_updates

    def _read_touch_once(self, client) -> dict[str, np.ndarray]:
        """Read all touch fields for one hand."""
        touch_updates: dict[str, np.ndarray] = {}
        for touch_key, segments in self._get_touch_segments(self.RegAddr).items():
            values = self._read_touch_field(client, segments)
            if values is not None:
                touch_updates[touch_key] = values
        return touch_updates

    def _refresh_sensors_once(self):
        """Read both state and touch once, synchronized to the control loop."""
        if self.hand_side == 'b':
            left_state = self._read_joint_states_once(self.client_left)
            left_touch = self._read_touch_once(self.client_left)
            right_state = self._read_joint_states_once(self.client_right)
            right_touch = self._read_touch_once(self.client_right)

            with self._data_lock:
                if left_state:
                    self.states['Left'].update(left_state)
                if left_touch:
                    self.touch['Left'].update(left_touch)
                if right_state:
                    self.states['Right'].update(right_state)
                if right_touch:
                    self.touch['Right'].update(right_touch)
        else:
            hand = 'Left' if self.hand_side == 'l' else 'Right'
            state = self._read_joint_states_once(self.client)
            touch = self._read_touch_once(self.client)
            with self._data_lock:
                if state:
                    self.states[hand].update(state)
                if touch:
                    self.touch[hand].update(touch)

    def read(self) -> Dict:
        """Return a thread-safe snapshot of state and touch data."""
        with self._data_lock:
            return {
                'states': {
                    hand: {key: value.copy() for key, value in hand_states.items()}
                    for hand, hand_states in self.states.items()
                },
                'touch': {
                    hand: {key: value.copy() for key, value in hand_touch.items()}
                    for hand, hand_touch in self.touch.items()
                },
            }

    # --- Control loop ---
    def _actuator_worker(self):
        """Background thread to continuously send Modbus commands without blocking."""
        send_interval = 0.02  # 50 Hz is plenty for the hand

        while self._running:
            try:
                if self._is_engaged:
                    if self.hand_side == 'b':
                        self.set_position(self.q_goal_left_hand, hand='l')
                        self.set_position(self.q_goal_right_hand, hand='r')
                    elif self.hand_side == 'l':
                        self.set_position(self.q_goal_left_hand)
                    else:  # 'r'
                        self.set_position(self.q_goal_right_hand)

                if self._sync_sensors_with_actuation:
                    self._refresh_sensors_once()
            except Exception as e:
                print(f"Error in synchronized control loop: {e}")

            time.sleep(send_interval)

    def _convert_rad_goals_to_hw_goals(self, finger_goals) -> list[int]:
        """Map radian finger goals (session packing order) to hw position units.

        Hardware position control expects integers in [0, 2000] per joint.
        """
        return [int(np.clip(finger_goals[i] / self.range_list[i] * 2000, 0, 2000))
                for i in range(6)]

    def set_joint_goals(self, goals):
        """Set target finger angles.

        Args:
            goals: A dict with ``q_goal_right_hand`` / ``q_goal_left_hand``
                keys (radians, 6 per hand) OR any object exposing the same
                names as attributes — e.g. ``geo_kin_core.types.RetargetOutput``.
                ``None`` (or a missing key/attribute) means "keep previous
                goal" for that hand.
        """
        if goals is None:
            return
        if isinstance(goals, Mapping) and not goals:
            return

        q_right = _get_goal(goals, "q_goal_right_hand")
        if q_right is not None:
            self.q_goal_right_hand = self._convert_rad_goals_to_hw_goals(
                np.asarray(q_right, dtype=float))

        q_left = _get_goal(goals, "q_goal_left_hand")
        if q_left is not None:
            self.q_goal_left_hand = self._convert_rad_goals_to_hw_goals(
                np.asarray(q_left, dtype=float))

    def apply_control(self, engaged=True):
        """Engage/disengage the background goal-streaming loop."""
        self._is_engaged = engaged
        return True

    # --- Low-level command modes ---
    def _send_command(self, mode, hand=None, **kwargs):
        """Route a command down via direct Modbus register writes."""
        if self.hand_side == 'b':
            if hand == 'l':
                client = self.client_left
            elif hand == 'r':
                client = self.client_right
            else:
                raise ValueError("hand parameter must be 'l' or 'r' when hand_side is 'b'")
        else:
            client = self.client

        try:
            with self._get_io_lock(client):
                if mode & self.ActStates.ANGLE:
                    client.write_registers(self.RegAddr.ANGLE_SET,
                                           kwargs.get('angle_set', [0] * 6), self.device_id)
                if mode & self.ActStates.POSITION:
                    client.write_registers(self.RegAddr.POS_SET,
                                           kwargs.get('pos_set', [0] * 6), self.device_id)
                if mode & self.ActStates.FORCE:
                    client.write_registers(self.RegAddr.FORCE_SET,
                                           kwargs.get('force_set', [0] * 6), self.device_id)
                if mode & self.ActStates.VELOCITY:
                    client.write_registers(self.RegAddr.SPEED_SET,
                                           kwargs.get('speed_set', [0] * 6), self.device_id)
            return True
        except Exception as e:
            print(f"Modbus write exception: {e}")
            return False

    def set_angle(self, angles):
        """Set raw joint angles ([pinky, ring, middle, index, thumb-bend, thumb-rot])."""
        if len(angles) != 6:
            raise ValueError("angles must be a list of 6 values")
        return self._send_command(self.ActStates.ANGLE, angle_set=angles)

    def set_position(self, positions, hand=None):
        """Set raw joint positions (0-2000 hw units, goals packing order)."""
        return self._send_command(mode=self.ActStates.POSITION, hand=hand,
                                  pos_set=positions)

    def set_force(self, forces):
        """Set force control values (goals packing order)."""
        if len(forces) != 6:
            raise ValueError("forces must be a list of 6 values")
        return self._send_command(mode=self.ActStates.FORCE, force_set=forces)

    def set_velocity(self, velocities):
        """Set joint velocities/speeds (goals packing order)."""
        if len(velocities) != 6:
            raise ValueError("velocities must be a list of 6 values")
        return self._send_command(mode=self.ActStates.VELOCITY, speed_set=velocities)

    def set_angle_and_position(self, angles, positions):
        """Set both angles and positions simultaneously."""
        if len(angles) != 6 or len(positions) != 6:
            raise ValueError("angles and positions must each be lists of 6 values")
        return self._send_command(mode=self.ActStates.ANGLE | self.ActStates.POSITION,
                                  angle_set=angles, pos_set=positions)

    def set_angle_and_velocity(self, angles, velocities):
        """Set both angles and velocities simultaneously."""
        if len(angles) != 6 or len(velocities) != 6:
            raise ValueError("angles and velocities must each be lists of 6 values")
        return self._send_command(mode=self.ActStates.ANGLE | self.ActStates.VELOCITY,
                                  angle_set=angles, speed_set=velocities)

    def stop(self):
        """Disengage, stop the worker thread, and send a no-op (IDLE) command."""
        self._is_engaged = False
        self._running = False
        if getattr(self, '_actuator_thread', None) is not None and self._actuator_thread.is_alive():
            self._actuator_thread.join(timeout=0.1)

        if self.hand_side == 'b':
            success_left = self._send_command(mode=self.ActStates.IDLE, hand='l')
            success_right = self._send_command(mode=self.ActStates.IDLE, hand='r')
            return success_left and success_right
        return self._send_command(mode=self.ActStates.IDLE)
