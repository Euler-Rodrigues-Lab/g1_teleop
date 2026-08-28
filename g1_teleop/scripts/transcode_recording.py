# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Transcode a recorded OpenXR CSV into a device-neutral geo_kin_core frame stream.

The CSV format needs the monolith's device stack (pandas + xr_robot_teleop_server
bone schemas); a frame stream is plain numpy and replays anywhere. Run this once
to produce sample/regression data, then everything downstream — the offline
replay demo, tests, public CI — reads the .npz and needs no device deps at all.

Example::

    python -m g1_teleop.scripts.transcode_recording \\
        --csv_file $GEO_TELEOP_MONOLITH/References/recordings/picking_up_mustard.csv \\
        --out g1_teleop/assets/sample_motion/picking_up_mustard.npz \\
        --fps 60 --duration 6
"""

import argparse

from geo_kin_core.frames import save_frames

from g1_teleop.input import OfflineCSVAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Recorded CSV -> geo_kin_core frame stream")
    parser.add_argument("--csv_file", required=True, help="Recorded OpenXR body-pose CSV")
    parser.add_argument("--out", required=True, help="Output .npz frame stream")
    parser.add_argument("--fps", type=float, default=60.0, help="Sampling rate (Hz)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time (s)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Seconds to transcode (default: to the end)")
    parser.add_argument("--monolith_path", default=None,
                        help="SEW-Geometric-Teleop checkout (else GEO_TELEOP_MONOLITH)")
    parser.add_argument("--notes", default="", help="Free-text note stored in the stream")
    return parser.parse_args()


def main():
    args = parse_args()
    source = OfflineCSVAdapter(args.csv_file, loop=False, monolith_path=args.monolith_path)
    duration = args.duration if args.duration is not None else source.duration - args.start
    n = max(int(round(duration * args.fps)), 0)

    frames = []
    for i in range(n):
        frame, _ = source.get_frame_at_time(args.start + i / args.fps)
        if frame is None:
            break
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No frames read from {args.csv_file} at start={args.start}s")

    path = save_frames(
        args.out, frames, fps=args.fps,
        source=f"{__import__('os').path.basename(args.csv_file)} "
               f"[{args.start:.2f}s +{len(frames) / args.fps:.2f}s @ {args.fps:g}Hz]",
        notes=args.notes,
    )
    size_mb = path.stat().st_size / 1e6
    print(f"Wrote {len(frames)} frames ({len(frames) / args.fps:.2f}s) to {path} [{size_mb:.2f} MB]")


if __name__ == "__main__":
    main()
