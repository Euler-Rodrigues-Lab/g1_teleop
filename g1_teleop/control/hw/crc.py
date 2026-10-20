# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Unitree ``LowCmd`` CRC without the SDK's bundled native library.

On Linux ``unitree_sdk2py.utils.crc.CRC`` loads ``utils/lib/crc_<arch>.so``, but
the SDK's setup.py does not package that directory, so pip / uv installs from
Git fail with ``crc_amd64.so: cannot open shared object file`` on the first
command. This computes the same CRC32 (poly 0x04C11DB7, init 0xFFFFFFFF, MSB
first over the message's little-endian 32-bit words, trailing crc word
excluded) through ``binascii``; it matches the SDK's pure-Python ``_crc_py``.
"""

import binascii
import struct

# Byte bit-reversal: the non-reflected CRC equals the reflected zlib CRC of
# bit-reversed bytes, bit-reversed back (without zlib's final xor).
_REV8 = bytes(int(f"{b:08b}"[::-1], 2) for b in range(256))

# Packing layouts copied from unitree_sdk2py.utils.crc.CRC.
_HG_LOWCMD = struct.Struct("<2B2x" + "B3x5fI" * 35 + "5I")
_GO_LOWCMD = struct.Struct("<4B4IH2x" + "B3x5f3I" * 20 + "4B" + "55Bx2I")


def crc32_words(raw: bytes) -> int:
    """Unitree CRC32 of ``raw`` read as little-endian 32-bit words."""
    n = len(raw) // 4
    big_endian = struct.pack(f">{n}I", *struct.unpack(f"<{n}I", raw[: 4 * n]))
    crc = binascii.crc32(big_endian.translate(_REV8)) ^ 0xFFFFFFFF
    return int(f"{crc:032b}"[::-1], 2)


def lowcmd_crc(cmd) -> int:
    """CRC for a ``unitree_hg`` or ``unitree_go`` ``LowCmd_`` (as ``CRC().Crc``)."""
    typename = cmd.__idl_typename__
    if typename == "unitree_hg.msg.dds_.LowCmd_":
        values = [cmd.mode_pr, cmd.mode_machine]
        for m in cmd.motor_cmd[:35]:
            values += [m.mode, m.q, m.dq, m.tau, m.kp, m.kd, m.reserve]
        raw = _HG_LOWCMD.pack(*values, *cmd.reserve, cmd.crc)
    elif typename == "unitree_go.msg.dds_.LowCmd_":
        values = [*cmd.head, cmd.level_flag, cmd.frame_reserve, *cmd.sn, *cmd.version, cmd.bandwidth]
        for m in cmd.motor_cmd[:20]:
            values += [m.mode, m.q, m.dq, m.tau, m.kp, m.kd, *m.reserve]
        values += [cmd.bms_cmd.off, *cmd.bms_cmd.reserve, *cmd.wireless_remote, *cmd.led,
                   *cmd.fan, cmd.gpio, cmd.reserve, cmd.crc]
        raw = _GO_LOWCMD.pack(*values)
    else:
        raise TypeError(f"no LowCmd CRC for {typename}")
    return crc32_words(raw[:-4])
