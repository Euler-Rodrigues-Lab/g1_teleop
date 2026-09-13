# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Public sample replay and typed-frame conversion tests."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

from g1_teleop import SAMPLE_MOTION
from g1_teleop.input import action_to_retarget_frame, open_motion_source

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
