# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Offline replay: the vendored sample frame stream, the adapter conversion, and
(when a monolith checkout with the device deps is available) the CSV path.

Only the CSV tests need the monolith. Everything else — including replaying the
vendored sample motion through the demo — runs on a clean checkout.
"""

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

from g1_teleop import SAMPLE_MOTION
from g1_teleop.input import action_to_retarget_frame, open_motion_source

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


def _args(*argv):
    import sys
    from g1_teleop.demos.replay_offline import parse_args

    old, sys.argv = sys.argv, ["replay_offline", *argv]
    try:
        return parse_args()
    finally:
        sys.argv = old


def test_sample_motion_is_vendored():
    """The demo must run on a clean checkout: no device, no monolith."""
    stream = open_motion_source(frames=SAMPLE_MOTION)
    assert stream.duration > 1.0
    frame = stream.frame_at_time(0.5)
    assert frame is not None
    assert frame.left_sew is not None and frame.right_sew is not None
    assert frame.R_lower_upper is not None          # torso solvable
    assert frame.left_fingers and frame.right_fingers  # hands solvable
    # exact joint key names the solvers look up
    assert "index_finger_mcp" in frame.right_fingers["index"]


@pytest.mark.geo
@pytest.mark.parametrize("hand", ["inspire", "psyonic"])
def test_replay_demo_headless_from_sample(hand):
    """Drive the demo's own build/step helpers over the vendored sample."""
    from g1_teleop.demos.replay_offline import build, count_self_contacts, step

    args = _args("--hand", hand, "--headless", "--no-loop")
    model, data, controller, session, source = build(args)
    solved = 0
    for i in range(20):
        frame, solve_time = step(model, data, controller, session, source, i / args.max_fr)
        if frame is not None:
            solved += 1
            assert solve_time >= 0.0
            assert count_self_contacts(model, data) >= 0
    assert solved == 20
    # The replay actually moved the robot off its initial pose.
    assert np.linalg.norm(data.qpos) > 0.0
    assert controller.q_goal_torso is not None


@needs_recording
@pytest.mark.geo
def test_replay_demo_headless_from_csv():
    """The CSV path stays wired (needs the monolith device stack)."""
    from g1_teleop.demos.replay_offline import build, step

    args = _args("--csv_file", str(RECORDING), "--headless", "--no-loop")
    model, data, controller, session, source = build(args)
    frame, _ = step(model, data, controller, session, source, 0.5)
    assert frame is not None
    assert controller.q_goal_torso is not None


def test_sample_motion_carries_the_capture_skeleton():
    """The overlay draws raw bones, so the vendored stream must ship them."""
    frame = open_motion_source(frames=SAMPLE_MOTION).frame_at_time(0.5)
    skeleton = frame.skeleton
    assert skeleton is not None
    names, parents = skeleton["names"], skeleton["parents"]
    assert len(names) == len(parents) == len(skeleton["positions"])
    assert (parents >= 0).sum() >= 60          # full-body skeleton, not a stub
    # the parts that were missing when the overlay reconstructed from SEW only
    assert any("Spine" in n for n in names)
    assert any("FootBall" in n for n in names)          # toes
    assert sum("Index" in n for n in names) >= 8        # per-hand finger bones


@needs_recording
def test_skeleton_matches_the_capture_side_visualizer():
    """Our overlay must draw exactly the segments the monolith viewer draws."""
    import sys

    import geo_kin_core.viz.capsules as caps
    from geo_kin_core.viz import HumanCapsuleViz
    from g1_teleop.input import OfflineCSVAdapter

    sys.path.insert(0, MONOLITH)
    from projects.shared_scripts.mujoco_human_capsule import MujocoHumanCapsule

    frame, bones = OfflineCSVAdapter(RECORDING, loop=False).get_frame_at_time(1.0)
    p_off = np.array([0.3, -0.2, 0.1])
    r_off = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    class FakeViewer:
        user_scn = object()

    reference = MujocoHumanCapsule(FakeViewer())
    ref_caps = []
    reference._add_capsule = lambda a, b, r: ref_caps.append(
        (tuple(np.round(a, 9)), tuple(np.round(b, 9)), round(r, 9)))
    reference.set_base_offset(p_off, r_off)
    reference.update(bones)

    ours = []
    original = caps.add_capsule
    caps.add_capsule = lambda v, a, b, r, rgba=None: (
        ours.append((tuple(np.round(a, 9)), tuple(np.round(b, 9)), round(r, 9))), True)[1]
    try:
        viz = HumanCapsuleViz(FakeViewer())
        viz.set_base_offset(p_off, r_off)
        viz.draw(frame)
    finally:
        caps.add_capsule = original

    assert len(ours) == len(ref_caps) > 60
    assert sorted(ours) == sorted(ref_caps)
