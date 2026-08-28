# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""G1 XR teleoperation demo (sim + optional real-robot DDS control).

Port of the monolith demo_g1_xr_robot_teleop_hw.py (arms/waist) and
demo_g1_inspire_xr_robot_teleop_full_body[_hw].py (inspire hands) onto the
geo_kin_core interface: solver = ``resolve_session(robot='g1', hand=...)``
(licensed `geo_kin` wheel -> private geo_kin_ref -> public mink fallback),
controller = :class:`g1_teleop.control.G1FullBodyMuJoCoController` (psyonic)
or :class:`g1_teleop.control.G1InspireHandMuJoCoController` (inspire-mounted
MJCF, hands actuated from q_goal_*_hand), device =
:class:`g1_teleop.input.XRDeviceAdapter` (interim; needs a monolith checkout
via --monolith_path or GEO_TELEOP_MONOLITH until the xrt_device repo exists).

In --hw mode the arms + 3-DOF waist go over Unitree DDS
(send_upper_body_targets) and, for --hand inspire, the finger goals go to the
Inspire hands over direct Modbus TCP (control.hw.InspireHandController).

Examples:
    # Sim only (viewer), inspire-mounted model, fingers actuated in sim
    python -m g1_teleop.demos.teleop_xr --hand inspire

    # Real robot over DDS (arms + waist) + inspire hands over Modbus
    python -m g1_teleop.demos.teleop_xr --hand inspire --hw --net eno1

    # Real robot over DDS (arms + waist), psyonic hands in sim
    python -m g1_teleop.demos.teleop_xr --hand psyonic --hw --net eno1

    # Hardware code path without sending commands
    python -m g1_teleop.demos.teleop_xr --hw --dry_run
"""

import argparse
import sys
import time
import traceback

import mujoco
import mujoco.viewer
import numpy as np

from geo_kin_core.session import resolve_session
from geo_kin_core.viz import HumanCapsuleViz, capsules, draw_filtered_sew

from g1_teleop import XML_INSPIRE_MOUNTED, XML_POSITION_CTRL_DANCE_W_HANDS
from g1_teleop.control import G1FullBodyMuJoCoController, G1InspireHandMuJoCoController
from g1_teleop.input import XRDeviceAdapter
from g1_teleop.mjcf import load_mjcf


def update_sim_from_hardware(controller, data, q_right, q_left, q_torso):
    """Update MuJoCo simulation state from hardware feedback."""
    if q_right is not None:
        for i, addr in enumerate(controller.right_arm_qpos_addrs):
            if i < len(q_right):
                data.qpos[addr] = q_right[i]
    if q_left is not None:
        for i, addr in enumerate(controller.left_arm_qpos_addrs):
            if i < len(q_left):
                data.qpos[addr] = q_left[i]
    if q_torso is not None:
        for i, addr in enumerate(controller.torso_qpos_addrs):
            if i < len(q_torso):
                data.qpos[addr] = q_torso[i]


def visualize_filtered_capsules(viewer, session, controller, rgba=(0.2, 0.9, 0.2, 0.35)):
    """Draw the post-XPBD filtered SEW capsules (solver diagnostics).

    Thin wrapper over :func:`geo_kin_core.viz.draw_filtered_sew` so every robot
    repo shares one implementation.
    """
    return draw_filtered_sew(viewer, session, to_world=controller.get_sew_transform(), rgba=rgba)


def parse_args():
    parser = argparse.ArgumentParser(description="G1 XR teleoperation (sim + --hw)")
    # Solver / embodiment
    parser.add_argument("--hand", choices=["inspire", "psyonic"], default="inspire",
                        help="Hand embodiment used for finger retargeting")
    parser.add_argument("--no-torso", dest="no_torso", action="store_true",
                        help="Disable the 3-DOF waist solve (arms only)")
    parser.add_argument("--no_safety_filter", action="store_true",
                        help="Disable the XPBD self-collision SEW filter")
    # Rates
    parser.add_argument("--max_fr", default=1000, type=int,
                        help="Maximum frame rate (sim step rate)")
    parser.add_argument("--ik_rate", default=60, type=int, help="IK update rate (Hz)")
    # Mocap scaling (monolith demo defaults)
    parser.add_argument("--shoulder_width_scale", type=float, default=0.8)
    parser.add_argument("--hip_width_scale", type=float, default=1.3)
    parser.add_argument("--mocap_scale", type=float, nargs=3, default=(0.85, 0.85, 0.65),
                        metavar=("SX", "SY", "SZ"),
                        help="Cartesian human->robot mocap scale")
    parser.add_argument("--mocap_offset", type=float, nargs=3, default=(0.0, 0.0, 0.4),
                        metavar=("OX", "OY", "OZ"), help="Mocap offset (m)")
    # Device (interim monolith import)
    parser.add_argument("--monolith_path", type=str, default=None,
                        help="SEW-Geometric-Teleop checkout for the XR device "
                             "(or set GEO_TELEOP_MONOLITH); temporary until "
                             "the xrt_device repo exists")
    parser.add_argument("--record_data", action="store_true",
                        help="Enable CSV recording of body pose and action data")
    parser.add_argument("--output_dir", type=str, default="./recordings")
    parser.add_argument("--ndigits", type=int, default=3)
    # Hardware
    parser.add_argument("--hw", action="store_true",
                        help="Send targets to the real G1 over Unitree DDS")
    parser.add_argument("--dry_run", action="store_true",
                        help="Do not initialize/send hardware commands (sim + IK only)")
    parser.add_argument("--net", type=str, default="enx00051b93f66a",
                        help="Network interface used by Unitree DDS (e.g., eno1)")
    parser.add_argument("--hw_kp_scale", type=float, default=1.0,
                        help="Scale for hardware joint kp")
    parser.add_argument("--hw_kd_scale", type=float, default=1.0,
                        help="Scale for hardware joint kd")
    parser.add_argument("--startup_time", type=float, default=2.0,
                        help="Seconds to interpolate robot to default pose at startup/home")
    parser.add_argument("--no_hw_startup", action="store_true",
                        help="Skip startup interpolation to deploy default pose")
    parser.add_argument("--no_exit_damping", action="store_true",
                        help="Skip sending damping command on exit")
    return parser.parse_args()


def main():
    args = parse_args()
    use_hw = args.hw and not args.dry_run

    # Sim model + controller per hand embodiment: psyonic hands are wired into
    # the dance_w_hands MJCF; inspire runs on the inspire-mounted MJCF
    # (signature-locked to the monolith layout — load_mjcf repoints its
    # meshdir at assets/meshes in-memory).
    if args.hand == "psyonic":
        xml = XML_POSITION_CTRL_DANCE_W_HANDS
        controller_cls = G1FullBodyMuJoCoController
    else:
        xml = XML_INSPIRE_MOUNTED
        controller_cls = G1InspireHandMuJoCoController
    print(f"Loading model from: {xml}")
    model = load_mjcf(xml)
    data = mujoco.MjData(model)

    print("Initializing teleoperation system...")
    try:
        device = XRDeviceAdapter(
            monolith_path=args.monolith_path,
            record_data=args.record_data,
            output_dir=args.output_dir,
            ndigits=args.ndigits,
        )

        session = resolve_session(
            robot="g1",
            hand=args.hand,
            control_rate_hz=float(args.ik_rate),
            elbow_filter_cutoff_hz=2.0,
            collision_avoidance=not args.no_safety_filter,
            torso=not args.no_torso,
            preprocess=dict(
                shoulder_width_scale=args.shoulder_width_scale,
                hip_width_scale=args.hip_width_scale,
                mocap_cartesian_scale=tuple(args.mocap_scale),
                mocap_offset=tuple(args.mocap_offset),
            ),
        )
        print(f"Solver backend: {type(session).__module__}.{type(session).__name__}")

        controller = controller_cls(model, data, debug=False)
        print("Teleoperation system initialized successfully!")
    except Exception as e:
        print(f"Error initializing teleoperation system: {e}")
        traceback.print_exc()
        sys.exit(1)

    hw_controller = None
    hand_hw_controller = None
    if use_hw:
        try:
            from g1_teleop.control.hw import Config, Controller, init_channel_factory

            print("Initializing real G1 hardware interface...")
            init_channel_factory(0, args.net)
            hw_controller = Controller(Config())
            print("Connected to real G1 low-level DDS.")
            if args.hand == "inspire":
                from g1_teleop.control.hw import InspireHandController

                print("Initializing Inspire hand hardware interface...")
                hand_hw_controller = InspireHandController(hand_side="b")
            # Sync simulation to current robot state immediately
            q_torso_hw, q_left_hw, q_right_hw = hw_controller.get_current_upper_body()
            update_sim_from_hardware(controller, data, q_right_hw, q_left_hw, q_torso_hw)
            mujoco.mj_forward(model, data)
        except Exception as e:
            print(f"Error initializing real G1 hardware: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Hardware disabled (run with --hw, without --dry_run, to enable).")

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=True,
    ) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -10
        viewer.cam.lookat[:] = [0, 0, 0.8]

        print("\nWaiting for a WebRTC client to connect...")
        while not device.is_connected and viewer.is_running():
            if hw_controller is not None and not args.no_hw_startup:
                hw_controller.move_to_default_pos()
                q_torso_hw, q_left_hw, q_right_hw = hw_controller.get_current_upper_body()
                update_sim_from_hardware(controller, data, q_right_hw, q_left_hw, q_torso_hw)
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.05)

        if not viewer.is_running():
            print("Viewer closed before client connection.")
            sys.exit(0)

        print("Client connected! Starting control loop.")
        viewer.sync()

        viz_interval = 1.0 / 30.0
        last_viz_time = 0.0
        ik_update_interval = 1.0 / args.ik_rate
        last_ik_update_time = 0.0

        try:
            while viewer.is_running():
                start_time = time.time()

                # Sync simulation with real robot state
                if hw_controller is not None:
                    try:
                        q_torso_hw, q_left_hw, q_right_hw = hw_controller.get_current_upper_body()
                        update_sim_from_hardware(controller, data, q_right_hw, q_left_hw, q_torso_hw)
                    except Exception as e:
                        print(f"Error reading hardware state: {e}")

                # --- Capture & solve IK ---
                if time.time() - last_ik_update_time >= ik_update_interval:
                    frame = device.get_frame()
                    if frame is not None:
                        # TODO: feed q_current_* back from hardware state for
                        # collision-avoidance anchoring (monolith left this
                        # None as well).
                        out = session.solve(frame, engaged=True,
                                            q_current_right=None, q_current_left=None)
                        controller.set_joint_goals(out)
                        if hand_hw_controller is not None:
                            # q_goal_*_hand -> inspire hw (same duck-typed contract)
                            hand_hw_controller.set_joint_goals(out)
                    last_ik_update_time = time.time()

                controller.apply_control(engaged=True)
                if hand_hw_controller is not None:
                    hand_hw_controller.apply_control(engaged=True)

                # --- Hardware send path (arms + 3-DOF waist) ---
                if hw_controller is not None:
                    q_goal_right = getattr(controller, "q_goal_right", None)
                    q_goal_left = getattr(controller, "q_goal_left", None)
                    q_goal_torso = getattr(controller, "q_goal_torso", None)
                    if (q_goal_right is not None and q_goal_left is not None
                            and q_goal_torso is not None):
                        hw_controller.send_upper_body_targets(
                            q_goal_torso=q_goal_torso,
                            q_goal_left=q_goal_left,
                            q_goal_right=q_goal_right,
                            kp_scale=args.hw_kp_scale,
                            kd_scale=args.hw_kd_scale,
                        )
                    # Inspire hands are driven by hand_hw_controller above.
                    # TODO: the psyonic serial hand controller is not ported
                    # yet; q_goal_*_hand is solved and available on the
                    # controller.

                # --- Visualization (filtered SEW capsules) ---
                if time.time() - last_viz_time >= viz_interval:
                    viewer.user_scn.ngeom = 0
                    visualize_filtered_capsules(viewer, session, controller)
                    last_viz_time = time.time()

                mujoco.mj_step(model, data)
                viewer.sync()

                elapsed = time.time() - start_time
                if elapsed < 1 / args.max_fr:
                    time.sleep(1 / args.max_fr - elapsed)
        except KeyboardInterrupt:
            pass

    print("Shutting down...")
    if args.record_data:
        device.cleanup()
    if hand_hw_controller is not None:
        try:
            hand_hw_controller.stop()
        except Exception as e:
            print(f"Error stopping Inspire hand hardware: {e}")
    if hw_controller is not None and not args.no_exit_damping:
        hw_controller.enter_damping(duration_s=0.3)
    print("Demo completed.")


if __name__ == "__main__":
    main()
