"""The public input and solver fallback must work without a private checkout."""
import sys
import numpy as np
import pytest


def test_shared_adapter_identity():
    from g1_teleop.input import XRDeviceAdapter, MediaPipeDeviceAdapter, OfflineCSVAdapter
    from xrt_devices.integrations import geo_kin
    assert XRDeviceAdapter is geo_kin.XRDeviceAdapter
    assert MediaPipeDeviceAdapter is geo_kin.MediaPipeDeviceAdapter
    assert OfflineCSVAdapter is geo_kin.OfflineCSVAdapter


def test_public_replay_fallback(monkeypatch):
    from g1_teleop.demos.replay_offline import parse_args, build, step
    monkeypatch.setitem(sys.modules, "geo_kin", None)
    monkeypatch.setitem(sys.modules, "geo_kin_ref", None)
    monkeypatch.setitem(sys.modules, "projects", None)
    monkeypatch.setattr(sys, "argv", ["replay", "--headless", "--backend", "mink"])
    model, data, controller, session, source = build(parse_args())
    for i in range(2):
        frame, _ = step(model, data, controller, session, source, i / 60)
        assert frame is not None
    assert np.isfinite(data.qpos).all()


def test_live_camera_arguments(monkeypatch):
    from g1_teleop.demos.teleop_xr import parse_args
    monkeypatch.setattr(sys, "argv", ["live", "--device", "mediapipe", "--backend", "mink"])
    args = parse_args()
    assert args.device == "mediapipe" and not args.hw


def test_live_device_closes_on_initialization_failure(monkeypatch):
    from g1_teleop.demos import teleop_xr
    closed = []
    class Device:
        def __init__(self, **kwargs):
            pass
        def cleanup(self):
            closed.append(True)
    def fail(args, device, resources):
        raise RuntimeError("solver initialization failed")
    monkeypatch.setattr(sys, "argv", ["live", "--device", "xrt"])
    monkeypatch.setattr(teleop_xr, "XRDeviceAdapter", Device)
    monkeypatch.setattr(teleop_xr, "run", fail)
    with pytest.raises(RuntimeError, match="solver initialization"):
        teleop_xr.main()
    assert closed == [True]
