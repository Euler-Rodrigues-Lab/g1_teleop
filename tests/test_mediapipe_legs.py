from types import SimpleNamespace as NS
import numpy as np
import pytest
from xrt_devices._mediapipe_legs import leg_targets


def pose():
    points = [NS(x=0., y=0., z=0.) for _ in range(33)]
    for side, x, ids in [('left', .12, (11,23,25,27,29,31)),
                         ('right', -.12, (12,24,26,28,30,32))]:
        for i, y, z in zip(ids, (-.5,0,.4,.8,.85,.85), (0,0,-.1,0,.03,-.15)):
            points[i] = NS(x=x,y=y,z=z)
    pixels = [NS(x=.5,y=.5,visibility=1.) for _ in points]
    return NS(pose_landmarks=NS(landmark=pixels), pose_world_landmarks=NS(landmark=points))


def test_measured_legs_and_occlusion():
    p = pose()
    output = leg_targets(p)
    for side in ('left','right'):
        leg = output[side+'_hka']
        assert leg is not None
        np.testing.assert_allclose(leg['ankle_rot'].T @ leg['ankle_rot'], np.eye(3), atol=1e-12)
        assert leg['A'][2] < leg['H'][2]
    p.pose_landmarks.landmark[25].visibility = .1
    assert leg_targets(p)['left_hka'] is None
    assert leg_targets(p)['right_hka'] is not None
    p.pose_landmarks.landmark[23].y = 1.2
    assert all(v is None for v in leg_targets(p).values())


@pytest.mark.geo
def test_leg_targets_reach_licensed_g1():
    geo_kin = pytest.importorskip('geo_kin')
    from xrt_devices.integrations.geo_kin import action_to_retarget_frame
    session = geo_kin.RetargetSession(robot='g1', hand=None, torso=False)
    result = session.solve(action_to_retarget_frame(leg_targets(pose())))
    for side in ('left','right'):
        q = getattr(result, 'q_goal_'+side+'_leg')
        assert q is not None and q.shape == (6,) and np.isfinite(q).all()

