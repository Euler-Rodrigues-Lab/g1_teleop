# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Replay bundled NPZ frames or public XRT CSV recordings in simulation."""

import argparse
import sys
import time
import traceback

import mujoco
import numpy as np

from g1_teleop.session import resolve_g1_session
from geo_kin_core.viz import HumanCapsuleViz, capsules, draw_filtered_sew

from g1_teleop import SAMPLE_MOTION, XML_INSPIRE_MOUNTED, XML_POSITION_CTRL_DANCE_W_HANDS
from g1_teleop.control import G1FullBodyMuJoCoController, G1InspireHandMuJoCoController
from g1_teleop.input import open_motion_source
from g1_teleop.mjcf import load_mjcf


def count_self_contacts(model, data) -> int:
    """Self-collision contacts in the current (kinematic) configuration.

    Contacts against static world geoms (floor) are ignored — the replayed
    robot is posed in the air, so only body-vs-body contacts are meaningful.
    """
    n = 0
    for i in range(data.ncon):
        con = data.contact[i]
        b1 = model.geom_bodyid[con.geom1]
        b2 = model.geom_bodyid[con.geom2]
        if b1 != 0 and b2 != 0 and b1 != b2:
            n += 1
    return n


def parse_args():
    parser = argparse.ArgumentParser(description="G1 offline CSV replay")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--frames", default=None,
                        help=f"geo_kin_core frame stream (.npz); default: the vendored "
                             f"sample motion ({SAMPLE_MOTION.name})")
    source.add_argument("--csv_file", default=None,
                        help="Recorded OpenXR body-pose CSV (xrt-devices[recording])")
    parser.add_argument("--hand", choices=["inspire", "psyonic"], default="inspire",
                        help="Hand embodiment (selects sim model + controller)")
    parser.add_argument("--playback_speed", type=float, default=1.0,
                        help="Playback speed multiplier (1.0 = real time)")
    parser.add_argument("--no-loop", dest="loop", action="store_false",
                        help="Stop at the end of the recording instead of looping")
    parser.add_argument("--max_fr", type=int, default=60,
                        help="Solve/step rate cap in Hz")
    parser.add_argument("--no-torso", dest="no_torso", action="store_true",
                        help="Disable the 3-DOF waist solve (arms only)")
    parser.add_argument("--no_safety_filter", action="store_true",
                        help="Disable the XPBD self-collision SEW filter")
    parser.add_argument("--base_alignment", choices=["manual", "mocap"], default="manual",
                        help="Base alignment mode. 'manual' (default, as in the monolith "
                             "offline demo) anchors the capture to the robot, so the human "
                             "overlay lands on it; 'mocap' leaves the human in raw capture "
                             "coordinates, away from the robot")
    parser.add_argument("--elbow_filter_hz", type=float, default=2.0,
                        help="Elbow-orientation low-pass cutoff (Hz)")
    parser.add_argument("--shoulder_width_scale", type=float, default=0.8)
    parser.add_argument("--hip_width_scale", type=float, default=1.3)
    parser.add_argument("--mocap_scale", type=float, nargs=3,
                        default=[0.85, 0.85, 0.65], metavar=("X", "Y", "Z"))
    parser.add_argument("--mocap_offset", type=float, nargs=3,
                        default=[0.0, 0.0, 0.4], metavar=("X", "Y", "Z"))
    parser.add_argument("--headless", action="store_true",
                        help="No viewer (timing/collision sweeps, CI)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Stop after this many solved frames")
    parser.add_argument("--log_stats", default=None,
                        help="Write per-frame solve time + contact count to this .npz")
    parser.add_argument("--no_human_overlay", action="store_true",
                        help="Skip the human-skeleton capsule overlay")
    parser.add_argument("--wall_clock", action="store_true",
                        help="Sample the recording by wall-clock time instead of a "
                             "fixed 1/max_fr step (non-reproducible if solving lags)")
    parser.add_argument("--backend", choices=["auto", "licensed", "reference", "mink"], default="auto")
    return parser.parse_args()


def build(args):
    """Model, data, controller, session, and playback source."""
    if args.hand == "psyonic":
        xml, controller_cls = XML_POSITION_CTRL_DANCE_W_HANDS, G1FullBodyMuJoCoController
    else:
        xml, controller_cls = XML_INSPIRE_MOUNTED, G1InspireHandMuJoCoController
    print(f"Loading model from: {xml}")
    model = load_mjcf(xml)
    data = mujoco.MjData(model)

    frames = args.frames
    if frames is None and args.csv_file is None:
        frames = SAMPLE_MOTION  # vendored sample: runs on a clean checkout
    source = open_motion_source(
        frames=frames,
        csv_file=args.csv_file,
        playback_speed=args.playback_speed,
        loop=args.loop,
    )
    print(f"Motion source: {source.describe()}")
    session = resolve_g1_session(model,
        backend=getattr(args, "backend", "auto"),
        hand=args.hand,
        control_rate_hz=float(args.max_fr),
        elbow_filter_cutoff_hz=args.elbow_filter_hz,
        collision_avoidance=not args.no_safety_filter,
        torso=not args.no_torso,
        base_alignment_mode=args.base_alignment,
        preprocess=dict(
            shoulder_width_scale=args.shoulder_width_scale,
            hip_width_scale=args.hip_width_scale,
            mocap_cartesian_scale=tuple(args.mocap_scale),
            mocap_offset=tuple(args.mocap_offset),
        ),
    )
    print(f"Solver backend: {type(session).__module__}.{type(session).__name__}")
    controller = controller_cls(model, data, debug=False)
    try:
        controller.setup_mocap_body("pelvis_mocap_mover")
    except Exception:
        pass  # model without a mocap mover body: robot just stays at its origin
    return model, data, controller, session, source


def step(model, data, controller, session, source, elapsed, engaged=True):
    """Solve + apply one replay frame. Returns ``(frame, solve_seconds)``."""
    frame = source.frame_at_time(elapsed)
    solve_time = 0.0
    if frame is not None:
        t0 = time.perf_counter()
        out = session.solve(frame, engaged=engaged)
        solve_time = time.perf_counter() - t0

        p_base = getattr(out, "p_world_base", None)
        R_base = getattr(out, "R_world_base", None)
        if p_base is not None and R_base is not None:
            controller.update_mocap_body(p_base, R_base)
        controller.set_joint_goals(out)

    controller.apply_control(engaged=engaged, kinematic_mode=True)
    mujoco.mj_forward(model, data)
    return frame, solve_time


def main():
    args = parse_args()
    try:
        model, data, controller, session, source = build(args)
    except Exception as e:
        print(f"Error initializing replay: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"Recording duration: {source.duration:.2f}s "
          f"(playback speed {args.playback_speed}x, loop={args.loop})")
    model.opt.timestep = 0.005
    solve_times, contacts = [], []
    frame_interval = 1.0 / args.max_fr

    def run(viewer=None):
        overlay = None
        if viewer is not None and not args.no_human_overlay:
            overlay = HumanCapsuleViz(viewer)
        start = time.time()
        sim_time = 0.0  # deterministic playback clock (see --wall_clock)
        n = 0
        finished = False
        while viewer is None or viewer.is_running():
            loop_start = time.time()
            elapsed = (loop_start - start) if args.wall_clock else sim_time
            sim_time += frame_interval * args.playback_speed
            try:
                frame, solve_time = step(model, data, controller, session, source, elapsed)
            except Exception as e:
                print(f"Error processing frame: {e}")
                traceback.print_exc()
                break

            if frame is None:
                if not args.loop and not finished:
                    print("Playback finished.")
                    finished = True
                    if viewer is None:
                        break
            else:
                n += 1
                solve_times.append(solve_time)
                contacts.append(count_self_contacts(model, data))

            if viewer is not None:
                capsules.clear(viewer)
                draw_filtered_sew(viewer, session, to_world=controller.get_sew_transform())
                if overlay is not None and frame is not None:
                    R_mocap = getattr(session, "R_mocap_world", None)
                    p_mocap = getattr(session, "p_mocap_world", None)
                    if R_mocap is not None and p_mocap is not None:
                        overlay.set_base_offset(p_mocap, np.asarray(R_mocap).T)
                    overlay.draw(frame)
                viewer.sync()

            if args.max_frames is not None and n >= args.max_frames:
                break
            lag = time.time() - loop_start
            if lag < frame_interval:
                time.sleep(frame_interval - lag)
        return n

    if args.headless:
        n = run(None)
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False,
        ) as viewer:
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -15
            viewer.cam.lookat[:] = [0, 0, 1.0]
            n = run(viewer)

    if solve_times:
        ms = np.asarray(solve_times) * 1e3
        print(f"Replayed {n} frames | solve {ms.mean():.2f} ms mean, "
              f"{ms.max():.2f} ms max | self-collision frames: "
              f"{int(np.count_nonzero(contacts))}/{len(contacts)}")
    if args.log_stats:
        np.savez(args.log_stats,
                 solve_time_s=np.asarray(solve_times),
                 self_contacts=np.asarray(contacts))
        print(f"Wrote stats to {args.log_stats}")


if __name__ == "__main__":
    main()
    print("Demo completed.")
