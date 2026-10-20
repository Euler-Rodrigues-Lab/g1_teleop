# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""LowCmd CRC must match Unitree's without the SDK's (unpackaged) crc_*.so."""

import ctypes
import random
import struct

import pytest

from g1_teleop.control.hw.crc import crc32_words, lowcmd_crc


def _unitree_crc_py(words):
    """Verbatim algorithm of unitree_sdk2py.utils.crc.CRC._crc_py."""
    crc, polynomial = 0xFFFFFFFF, 0x04C11DB7
    for current in words:
        bit = 1 << 31
        for _ in range(32):
            crc = ((crc << 1) & 0xFFFFFFFF) ^ (polynomial if crc & 0x80000000 else 0)
            if current & bit:
                crc ^= polynomial
            bit >>= 1
    return crc


@pytest.mark.parametrize("n_words", [1, 2, 7, 250])
def test_crc32_words_matches_unitree_reference(n_words):
    rng = random.Random(n_words)
    for _ in range(20):
        words = [rng.getrandbits(32) for _ in range(n_words)]
        assert crc32_words(struct.pack(f"<{n_words}I", *words)) == _unitree_crc_py(words)


def test_lowcmd_crc_matches_sdk_packing():
    sdk_crc = pytest.importorskip("unitree_sdk2py.utils.crc")
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

    class _NoLib:  # the SDK's CRC() loads crc_<arch>.so; force its pure-Python path
        crc32_core = type("fn", (), {})()

    real_cdll, ctypes.CDLL = ctypes.CDLL, lambda path: _NoLib()
    try:
        reference = sdk_crc.CRC()
    finally:
        ctypes.CDLL = real_cdll
    reference.platform = "Darwin"

    rng = random.Random(0)
    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr, cmd.mode_machine = 1, rng.randrange(256)
    for motor in cmd.motor_cmd:
        motor.mode = 1
        motor.q, motor.dq, motor.tau, motor.kp, motor.kd = (rng.uniform(-50, 50) for _ in range(5))
    assert lowcmd_crc(cmd) == reference.Crc(cmd)
