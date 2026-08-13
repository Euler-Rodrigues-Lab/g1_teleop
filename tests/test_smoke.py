# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Headless smoke tests: assets, MJCF load, controller, and (when the private
geo_kin_ref reference is installed) a 20-frame teleop loop through
resolve_session -> G1Session -> G1FullBodyMuJoCoController -> mj_step.

Synthetic-stream construction reuses the pattern proven by the geo_kin
equivalence suite (tests/test_equiv_g1_session.py): smooth reachable SEW
trajectories from FK on sinusoidal joint trajectories, plus wrist rotations,
finger keypoints, and a NON-IDENTITY R_lower_upper so the torso solve runs.
"""

import importlib.util
import os

# Headless: no GL context is created by these tests (no Renderer/viewer),
# but pin the EGL backend so any accidental render path stays off-screen.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g1_teleop import (
    SPECS_DIR,
    URDF_29DOF,
    XML_INSPIRE_MOUNTED,
    XML_POSITION_CTRL,
    XML_POSITION_CTRL_DANCE,
    XML_POSITION_CTRL_DANCE_W_HANDS,
)
from g1_teleop.control import G1FullBodyMuJoCoController
from g1_teleop.mjcf import load_mjcf

HAS_GEO_REF = importlib.util.find_spec("geo_kin_ref") is not None
HAS_LICENSED_WHEEL = importlib.util.find_spec("geo_kin") is not None

N_FRAMES = 20
ARM_JOINT_NAMES = {
    "right_arm": [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    "left_arm": [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
}

# Finger keypoint bases (body-centric, meters) — same synthetic layout the
# geo_kin equivalence suite drives both solver generations with.
_KP_BASES = {
    "index": {
        "index_finger_mcp": [0.08, 0.02, 0.0],
        "index_finger_pip": [0.10, 0.02, -0.01],
        "index_finger_dip": [0.11, 0.02, -0.015],
        "index_finger_tip": [0.12, 0.02, -0.02],
    },
    "middle": {
        "middle_finger_mcp": [0.08, 0.0, 0.0],
        "middle_finger_pip": [0.10, 0.0, -0.01],
        "middle_finger_dip": [0.11, 0.0, -0.015],
        "middle_finger_tip": [0.12, 0.0, -0.02],
    },
    "ring": {
        "ring_finger_mcp": [0.08, -0.02, 0.0],
        "ring_finger_pip": [0.10, -0.02, -0.01],
        "ring_finger_dip": [0.11, -0.02, -0.015],
        "ring_finger_tip": [0.12, -0.02, -0.02],
    },
    "pinky": {
        "pinky_mcp": [0.08, -0.04, 0.0],
        "pinky_pip": [0.10, -0.04, -0.01],
        "pinky_dip": [0.11, -0.04, -0.015],
        "pinky_tip": [0.12, -0.04, -0.02],
    },
    "thumb": {
        "thumb_cmc": [0.02, 0.02, 0.005],
        "thumb_mcp": [0.03, 0.03, 0.01],
        "thumb_pip": [0.05, 0.04, 0.02],
        "thumb_ip": [0.055, 0.042, 0.022],
        "thumb_tip": [0.07, 0.05, 0.03],
    },
}


# ---------------------------------------------------------------------------
# Assets: spec signatures + MJCF loadability
# ---------------------------------------------------------------------------

def test_spec_signatures_verify_against_copied_assets():
    """Committed spec npz sha256 must match the byte-identical asset copies."""
    from geo_kin_core.spec import verify_signature

    pairs = [
        (SPECS_DIR / "g1_29dof.npz", URDF_29DOF),
        (SPECS_DIR / "inspire_left.npz", XML_INSPIRE_MOUNTED),
        (SPECS_DIR / "inspire_right.npz", XML_INSPIRE_MOUNTED),
        (SPECS_DIR / "psyonic_left.npz", XML_POSITION_CTRL_DANCE_W_HANDS),
        (SPECS_DIR / "psyonic_right.npz", XML_POSITION_CTRL_DANCE_W_HANDS),
    ]
    for npz, src in pairs:
        assert npz.exists(), f"missing spec artifact {npz}"
        assert src.exists(), f"missing asset {src}"
        assert verify_signature(str(npz), str(src)), \
            f"{npz.name}: source sha256 drifted vs {src.name}"


@pytest.mark.parametrize("xml", [XML_POSITION_CTRL, XML_POSITION_CTRL_DANCE,
                                 XML_POSITION_CTRL_DANCE_W_HANDS],
                         ids=lambda p: p.name)
def test_mjcf_loads_and_steps(xml):
    # load_mjcf repairs the stale 'stand' keyframe the signature-locked dance
    # variants carry (see g1_teleop.mjcf docstring).
    model = load_mjcf(xml)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.time > 0.0


# ---------------------------------------------------------------------------
# Controller: duck-typed goals (dict AND attribute-object)
# ---------------------------------------------------------------------------

def test_controller_accepts_dict_and_object_goals():
    model = load_mjcf(XML_POSITION_CTRL_DANCE_W_HANDS)
    data = mujoco.MjData(model)
    controller = G1FullBodyMuJoCoController(model, data)
    assert len(controller.right_arm_actuator_ids) == 7
    assert len(controller.right_hand_actuator_ids) == 10
    assert len(controller.torso_actuator_ids) == 3

    # Legacy dict contract
    controller.set_joint_goals({"q_goal_right": np.full(7, 0.1), "q_goal_torso": None})
    assert np.allclose(controller.q_goal_right, 0.1)
    torso_before = controller.q_goal_torso.copy()  # None = keep previous

    # RetargetOutput-style attribute object (licensed wheel contract)
    from geo_kin_core.types import RetargetOutput

    out = RetargetOutput(q_goal_left=np.full(7, -0.2), q_goal_torso=np.array([0.1, 0.0, 0.05]))
    controller.set_joint_goals(out)
    assert np.allclose(controller.q_goal_left, -0.2)
    assert np.allclose(controller.q_goal_torso, [0.1, 0.0, 0.05])
    assert np.allclose(controller.q_goal_right, 0.1)  # untouched by None
    del torso_before

    controller.apply_control(engaged=True)
    mujoco.mj_step(model, data)
    assert data.time > 0.0


# ---------------------------------------------------------------------------
# XR adapter: legacy action dict -> RetargetFrame (pure conversion, no device)
# ---------------------------------------------------------------------------

def test_action_dict_to_retarget_frame():
    from g1_teleop.input import action_to_retarget_frame

    R_lu = Rotation.from_rotvec([0.1, -0.05, 0.2]).as_matrix()
    wrist = Rotation.from_euler("ZYX", [0.3, 0.1, -0.2]).as_matrix()
    action = {
        "R_world_upper_body": np.eye(3),
        "p_world_upper_body": np.array([0.0, 0.0, 1.2]),
        "left_sew": np.concatenate([[0, .2, 1.3], [0.1, .3, 1.1], [0.3, .3, 1.1],
                                    np.eye(3).flatten()]),
        "right_sew": np.concatenate([[0, -.2, 1.3], [0.1, -.3, 1.1], [0.3, -.3, 1.1],
                                     wrist.flatten()]),
        "head_rotation": np.eye(3),
        "left_fingers": {"index": {"index_finger_tip": np.zeros(3)}},
        "right_fingers": None,
        "R_lower_upper": R_lu,
        "body_center": np.array([0.0, 0.0, 0.9]),
        "ankle_to_body": np.array([0.02, 0.0, 0.9]),
        "left_hka": None,
        "right_hka": None,
        "left_gripper_val": 0.7,
        "right_gripper_val": 0.2,
    }
    frame = action_to_retarget_frame(action)
    assert np.allclose(frame.R_lower_upper, R_lu)  # torso path preserved
    assert np.allclose(frame.right_sew.W, [0.3, -.3, 1.1])
    assert np.allclose(frame.right_sew.R_world_wrist, wrist)
    assert frame.left_fingers is not None and frame.right_fingers is None
    assert frame.left_gripper_val == 0.7 and frame.right_gripper_val == 0.2
    assert np.allclose(frame.extras["body_center"], [0.0, 0.0, 0.9])
    assert "tags" not in frame.extras  # None/missing keys are dropped
    assert action_to_retarget_frame(None) is None
    assert action_to_retarget_frame({}) is None


# ---------------------------------------------------------------------------
# Synthetic stream (geo_kin_ref FK -> reachable SEW frames)
# ---------------------------------------------------------------------------

class SyntheticStream:
    """Deterministic smooth RetargetFrame generator (reachable via robot FK)."""

    def __init__(self, model, seed=20260812):
        from geo_kin_core.spec import load_spec
        from geo_kin_ref.g1 import arms as g1_arms

        self._fk = g1_arms.get_robot_SEW_from_q
        spec = load_spec(str(SPECS_DIR / "g1_29dof.npz"))
        self.transforms = {part: spec[part] for part in ("right_arm", "left_arm")}

        rng = np.random.default_rng(seed)
        self.arm_params = {}
        for part, names in ARM_JOINT_NAMES.items():
            lo = np.array([model.joint(n).range[0] for n in names])
            hi = np.array([model.joint(n).range[1] for n in names])
            c = (lo + hi) / 2.0
            a = 0.3 * (hi - lo) / 2.0
            f = rng.uniform(0.5, 2.5, size=len(names))
            phi = rng.uniform(0, 2 * np.pi, size=len(names))
            self.arm_params[part] = (c, a, f, phi)
        self.finger_params = {
            side: {
                finger: {k: (rng.uniform(0.005, 0.02, size=3),
                             rng.uniform(0.5, 3.0, size=3),
                             rng.uniform(0, 2 * np.pi, size=3))
                         for k in joints}
                for finger, joints in _KP_BASES.items()
            }
            for side in ("left", "right")
        }

    def _q_arm(self, part, t):
        c, a, f, phi = self.arm_params[part]
        return c + a * np.sin(2 * np.pi * f * t / N_FRAMES + phi)

    def _fingers(self, side, t):
        out = {}
        for finger, joints in _KP_BASES.items():
            fd = {}
            for k, base in joints.items():
                amp, f, phi = self.finger_params[side][finger][k]
                fd[k] = np.asarray(base, dtype=float) + amp * np.sin(
                    2 * np.pi * f * t / N_FRAMES + phi)
            out[finger] = fd
        return out

    def frame(self, t):
        from geo_kin_core.types import RetargetFrame, SEWPose

        sew = {}
        for part, side in (("right_arm", "right"), ("left_arm", "left")):
            fk = self._fk(self._q_arm(part, t), self.transforms[part])
            sew[side] = SEWPose(S=fk["S"].copy(), E=fk["E"].copy(), W=fk["W"].copy(),
                                R_world_wrist=fk["R_0_7"].copy())

        s = 2 * np.pi * t / N_FRAMES
        return RetargetFrame(
            left_sew=sew["left"],
            right_sew=sew["right"],
            R_world_upper_body=Rotation.from_euler(
                "ZYX", [0.3 * np.sin(0.9 * s), 0.1 * np.sin(1.7 * s + 0.5),
                        0.08 * np.sin(1.1 * s + 2.2)]).as_matrix(),
            p_world_upper_body=np.array(
                [0.10 * np.sin(0.8 * s), 0.08 * np.sin(1.2 * s + 0.7),
                 1.20 + 0.05 * np.sin(1.5 * s + 1.9)]),
            head_rotation=Rotation.from_euler(
                "ZYX", [0.25 * np.sin(1.1 * s + 0.3), -0.15 * np.sin(0.6 * s + 1.4),
                        0.05 * np.sin(1.8 * s)]).as_matrix(),
            # Non-identity so the 3-DOF waist solve actually runs.
            R_lower_upper=Rotation.from_rotvec(
                [0.20 * np.sin(1.3 * s + 0.4), 0.15 * np.sin(0.7 * s + 1.1),
                 0.30 * np.sin(1.9 * s + 2.0)]).as_matrix(),
            left_fingers=self._fingers("left", t),
            right_fingers=self._fingers("right", t),
            left_gripper_val=0.5 + 0.4 * np.sin(1.4 * s),
            right_gripper_val=0.5 + 0.4 * np.cos(0.9 * s),
        )


# ---------------------------------------------------------------------------
# End-to-end teleop smoke (needs the private geo_kin_ref backend)
# ---------------------------------------------------------------------------

@pytest.mark.geo
@pytest.mark.skipif(not HAS_GEO_REF,
                    reason="geo_kin_ref not installed (private dev machine only)")
def test_g1_session_teleop_smoke():
    from geo_kin_core.session import resolve_session

    model = mujoco.MjModel.from_xml_path(str(XML_POSITION_CTRL))
    data = mujoco.MjData(model)
    controller = G1FullBodyMuJoCoController(model, data)

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
    if not HAS_LICENSED_WHEEL:
        from geo_kin_ref.g1.session import G1Session

        assert isinstance(session, G1Session), \
            "resolve_session should fall through to the geo_kin_ref backend"

    session.reset(q_init_right=np.zeros(7), q_init_left=np.zeros(7))
    stream = SyntheticStream(model)
    q_right_start = controller.q_current_right.copy()

    for t in range(N_FRAMES):
        out = session.solve(stream.frame(t), engaged=True,
                            q_current_right=None, q_current_left=None)

        assert out.q_goal_right is not None and out.q_goal_right.shape == (7,), \
            f"frame {t}: right arm IK returned None"
        assert out.q_goal_left is not None and out.q_goal_left.shape == (7,), \
            f"frame {t}: left arm IK returned None"
        assert out.q_goal_torso is not None and np.asarray(out.q_goal_torso).shape == (3,), \
            f"frame {t}: torso (waist) solve returned None"
        assert np.all(np.isfinite(out.q_goal_right)) and np.all(np.isfinite(out.q_goal_left))
        assert out.q_goal_right_hand is not None and out.q_goal_left_hand is not None, \
            f"frame {t}: inspire hand IK returned None"

        controller.set_joint_goals(out)  # RetargetOutput object, not a dict
        controller.apply_control(engaged=True)
        mujoco.mj_step(model, data)

    assert data.time == pytest.approx(N_FRAMES * model.opt.timestep), \
        "mj_step did not advance simulation time"
    assert not np.allclose(controller.q_current_right, q_right_start), \
        "arm joints never moved under position control"
