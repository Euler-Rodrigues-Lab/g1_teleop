# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Headless inspire-hand smoke tests: inspire-mounted MJCF load (in-memory
meshdir repair), G1InspireHandMuJoCoController goal wiring, and (when the
private geo_kin_ref reference is installed) a 20-frame teleop loop through
resolve_session(robot='g1', hand='inspire') asserting the q_goal_*_hand
outputs flow into the Inspire hand actuators.

Synthetic frames reuse the SyntheticStream helper from test_smoke (itself the
pattern proven by the geo_kin equivalence suite, tests/test_equiv_g1_session.py).
"""

import importlib.util
import os

# Headless: no GL context is created by these tests (no Renderer/viewer),
# but pin the EGL backend so any accidental render path stays off-screen.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pytest

from g1_teleop import SPECS_DIR, XML_INSPIRE_MOUNTED
from g1_teleop.control import G1InspireHandMuJoCoController
from g1_teleop.mjcf import load_mjcf
from test_smoke import N_FRAMES, SyntheticStream

HAS_GEO_REF = importlib.util.find_spec("geo_kin_ref") is not None


def test_inspire_mjcf_loads_and_steps():
    # The committed XML is signature-locked with the monolith-layout meshdir;
    # load_mjcf repoints it at assets/meshes in-memory (see g1_teleop.mjcf).
    model = load_mjcf(XML_INSPIRE_MOUNTED)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.time > 0.0


def test_inspire_controller_hand_goal_wiring():
    model = load_mjcf(XML_INSPIRE_MOUNTED)
    data = mujoco.MjData(model)
    controller = G1InspireHandMuJoCoController(model, data)
    assert len(controller.right_arm_actuator_ids) == 7
    assert len(controller.right_hand_actuator_ids) == 6
    assert len(controller.left_hand_actuator_ids) == 6
    assert len(controller.torso_actuator_ids) == 3
    # Passive distal finger joints are coupled via joint equalities.
    assert len(controller.joint_equality_constraints) == 12
    # q_goal_*_hand[i] must drive the actuator of the i-th packing-order joint.
    assert [model.actuator(a).name for a in controller.right_hand_actuator_ids] == \
        controller.right_hand_joint_names

    # Legacy dict contract
    controller.set_joint_goals({"q_goal_right_hand": np.full(6, 0.3),
                                "q_goal_left_hand": None})
    assert np.allclose(controller.q_goal_right_hand, 0.3)

    # RetargetOutput-style attribute object (licensed wheel contract)
    from geo_kin_core.types import RetargetOutput

    out = RetargetOutput(q_goal_left_hand=np.full(6, 0.2))
    controller.set_joint_goals(out)
    assert np.allclose(controller.q_goal_left_hand, 0.2)
    assert np.allclose(controller.q_goal_right_hand, 0.3)  # None = keep previous

    controller.apply_control(engaged=True)
    assert np.allclose(data.ctrl[controller.right_hand_actuator_ids], 0.3)
    assert np.allclose(data.ctrl[controller.left_hand_actuator_ids], 0.2)
    mujoco.mj_step(model, data)
    assert data.time > 0.0


@pytest.mark.geo
@pytest.mark.skipif(not HAS_GEO_REF,
                    reason="geo_kin_ref not installed (private dev machine only)")
def test_g1_inspire_session_teleop_smoke():
    from geo_kin_core.session import resolve_session

    model = load_mjcf(XML_INSPIRE_MOUNTED)
    data = mujoco.MjData(model)
    controller = G1InspireHandMuJoCoController(model, data)

    session = resolve_session(
        robot="g1",
        hand="inspire",
        control_rate_hz=60.0,
        elbow_filter_cutoff_hz=2.0,
        collision_avoidance=True,
        torso=True,
        preprocess=dict(
            shoulder_width_scale=0.8,
            hip_width_scale=1.3,
            mocap_cartesian_scale=(0.85, 0.85, 0.65),
            mocap_offset=(0.0, 0.0, 0.4),
        ),
        spec_dir=SPECS_DIR,
    )
    session.reset(q_init_right=np.zeros(7), q_init_left=np.zeros(7))
    stream = SyntheticStream(model)

    right_hand_ctrl_start = data.ctrl[controller.right_hand_actuator_ids].copy()
    left_hand_ctrl_start = data.ctrl[controller.left_hand_actuator_ids].copy()

    for t in range(N_FRAMES):
        out = session.solve(stream.frame(t), engaged=True,
                            q_current_right=None, q_current_left=None)

        assert out.q_goal_right is not None and out.q_goal_right.shape == (7,), \
            f"frame {t}: right arm IK returned None"
        assert out.q_goal_left is not None and out.q_goal_left.shape == (7,), \
            f"frame {t}: left arm IK returned None"
        for side, q_hand in (("right", out.q_goal_right_hand),
                             ("left", out.q_goal_left_hand)):
            assert q_hand is not None, f"frame {t}: {side} inspire hand IK returned None"
            q_hand = np.asarray(q_hand)
            assert q_hand.shape == (6,), \
                f"frame {t}: {side} hand goal shape {q_hand.shape} != (6,)"
            assert np.all(np.isfinite(q_hand))

        controller.set_joint_goals(out)  # RetargetOutput object, not a dict
        controller.apply_control(engaged=True)
        mujoco.mj_step(model, data)

    assert data.time == pytest.approx(N_FRAMES * model.opt.timestep), \
        "mj_step did not advance simulation time"
    assert not np.allclose(data.ctrl[controller.right_hand_actuator_ids],
                           right_hand_ctrl_start), \
        "q_goal_right_hand never reached the Inspire hand actuators"
    assert not np.allclose(data.ctrl[controller.left_hand_actuator_ids],
                           left_hand_ctrl_start), \
        "q_goal_left_hand never reached the Inspire hand actuators"
