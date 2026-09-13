# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Transcode a public XRT CSV recording into a device-neutral NPZ frame stream."""

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
    parser.add_argument("--notes", default="", help="Free-text note stored in the stream")
    return parser.parse_args()


def main():
    args = parse_args()
    source = OfflineCSVAdapter(args.csv_file, loop=False)
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
