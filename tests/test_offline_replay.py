# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Offline CSV replay: adapter conversion + (when a monolith checkout with the
device deps is available) a real recording driven through the replay demo.

The conversion tests run everywhere — they need neither the monolith, nor a
recording, nor a solver backend.
"""

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

from g1_teleop.input import action_to_retarget_frame

MONOLITH = os.environ.get("GEO_TELEOP_MONOLITH")


def _monolith_recording():
    """A recording from the monolith checkout, if the device deps are present."""
    if not MONOLITH:
        return None
    if importlib.util.find_spec("pandas") is None:
        return None
    if importlib.util.find_spec("xr_robot_teleop_server") is None:
        return None
    for name in ("picking_up_mustard.csv", "fridge_stationary.csv"):
        csv = Path(MONOLITH) / "References" / "recordings" / name
        if csv.exists():
            return csv
    return None


RECORDING = _monolith_recording()
needs_recording = pytest.mark.skipif(
    RECORDING is None,
    reason="needs GEO_TELEOP_MONOLITH + device deps (pandas, xr_robot_teleop_server)",
)


def test_leg_keypoint_dicts_pass_through():
    """Regression: the device sends per-leg DICTS, not arrays, for *_hka.

    Coercing them to float arrays raised TypeError on the first real recorded
    frame (the monolith demos dodged it by nulling the keys before solving).
    """
    hka = {
        "H": np.zeros(3), "K": np.ones(3), "A": np.full(3, 2.0),
        "ankle_rot": np.eye(3), "A_world": np.zeros(3),
        "hip_center_world": np.zeros(3),
    }
    frame = action_to_retarget_frame({
        "left_sew": np.arange(18, dtype=float),
        "left_hka": hka,
        "right_hka": hka,
        "R_lower_upper": np.eye(3),
    })
    assert frame is not None
    assert frame.left_hka is hka and frame.right_hka is hka
    assert frame.R_lower_upper is not None  # torso still solvable


def test_empty_action_is_none():
    assert action_to_retarget_frame({}) is None
    assert action_to_retarget_frame(None) is None


@needs_recording
def test_offline_adapter_yields_frames():
    from g1_teleop.input import OfflineCSVAdapter

    src = OfflineCSVAdapter(RECORDING, loop=False)
    assert src.duration > 0.0
    frame, bones = src.get_frame_at_time(0.5)
    assert frame is not None and bones is not None
    assert frame.right_sew is not None and frame.left_sew is not None
    assert frame.R_lower_upper is not None


@needs_recording
@pytest.mark.geo
@pytest.mark.parametrize("hand", ["inspire", "psyonic"])
def test_replay_demo_headless(hand):
    """Drive the demo's own build/step helpers over a real recording."""
    from g1_teleop.demos.replay_offline import build, count_self_contacts, parse_args, step

    argv = ["--csv_file", str(RECORDING), "--hand", hand, "--headless", "--no-loop"]
    import sys
    old, sys.argv = sys.argv, ["replay_offline", *argv]
    try:
        args = parse_args()
    finally:
        sys.argv = old

    model, data, controller, session, source = build(args)
    solved = 0
    for i in range(20):
        bones, solve_time = step(model, data, controller, session, source, i / args.max_fr)
        if bones is not None:
            solved += 1
            assert solve_time >= 0.0
            assert count_self_contacts(model, data) >= 0
    assert solved == 20
    # The replay actually moved the robot off its initial pose.
    assert np.linalg.norm(data.qpos) > 0.0
    assert controller.q_goal_torso is not None
