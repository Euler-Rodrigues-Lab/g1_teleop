# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Full-body device frame -> geometric session -> MuJoCo leg joints."""
import mujoco
import numpy as np
import pytest

from g1_teleop import SAMPLE_MOTION, XML_INSPIRE_MOUNTED, XML_POSITION_CTRL_DANCE_W_HANDS
from g1_teleop.control import G1FullBodyMuJoCoController, G1InspireHandMuJoCoController
from g1_teleop.input import action_to_retarget_frame, open_motion_source
from g1_teleop.mjcf import load_mjcf
from g1_teleop.session import resolve_g1_session

pytestmark = pytest.mark.geo


@pytest.mark.parametrize("backend", ["licensed", "reference"])
@pytest.mark.parametrize("hand", ["inspire", "psyonic"])
def test_recorded_leg_targets_reach_mujoco(backend, hand):
    pytest.importorskip("geo_kin" if backend == "licensed" else "geo_kin_ref")
    xml, controller_type = ((XML_INSPIRE_MOUNTED, G1InspireHandMuJoCoController)
                            if hand == "inspire" else
                            (XML_POSITION_CTRL_DANCE_W_HANDS, G1FullBodyMuJoCoController))
    model = load_mjcf(xml)
    data = mujoco.MjData(model)
    controller = controller_type(model, data, debug=False)
    # Hand=None isolates legs and needs only the G1 robot license feature.
    session = resolve_g1_session(model, backend=backend, hand=None, control_rate_hz=60,
                                 preprocess={"hip_width_scale": 1.3})
    source = open_motion_source(frames=SAMPLE_MOTION)
    initial = {side: getattr(controller, f"q_goal_{side}_leg").copy()
               for side in ("right", "left")}
    for t in np.linspace(0, min(source.duration, 3.0), 20, endpoint=False):
        recorded = source.frame_at_time(t)
        # Use the public device adapter on a real recording's lower-body payload.
        frame = action_to_retarget_frame(dict(left_hka=recorded.left_hka,
                                               right_hka=recorded.right_hka,
                                               R_lower_upper=recorded.R_lower_upper))
        out = session.solve(frame)
        controller.set_joint_goals(out)
        controller.apply_control(engaged=True, kinematic_mode=True)
        mujoco.mj_forward(model, data)
        for side in ("right", "left"):
            q = getattr(out, f"q_goal_{side}_leg")
            assert q is not None and q.shape == (6,) and np.isfinite(q).all()
            np.testing.assert_array_equal(data.qpos[getattr(controller, f"{side}_leg_qpos_addrs")], q)
            controller._apply_position_control()
            np.testing.assert_array_equal(data.ctrl[getattr(controller, f"{side}_leg_actuator_ids")], q)
            assert out.diag.solution_branch[f"{side}_leg"] == 0
    for side in ("right", "left"):
        assert np.linalg.norm(getattr(controller, f"q_goal_{side}_leg") - initial[side]) > .01

    # Lost tracking holds that side while the other leg continues solving.
    held = controller.q_goal_right_leg.copy()
    frame.right_hka = None
    out = session.solve(frame)
    assert out.q_goal_right_leg is None and out.q_goal_left_leg is not None
    controller.set_joint_goals(out)
    np.testing.assert_array_equal(controller.q_goal_right_leg, held)
    held_left = controller.q_goal_left_leg.copy()
    controller.set_joint_goals(session.solve(frame, engaged=False))
    np.testing.assert_array_equal(controller.q_goal_left_leg, held_left)
