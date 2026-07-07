#!/usr/bin/env python3
"""Prepare a video for K230 board-side frame-sequence inference.

The board executable intentionally avoids decoding arbitrary video containers.
This helper uses host-side ffmpeg to convert a video into zero-padded image
frames that can be copied to the board and passed as input_mode.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a video into frames for falldown_detect.elf."
    )
    parser.add_argument("video", help="Input video path, for example falldown.avi")
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="Output frame directory. Default: <video_stem>_frames",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Output frame rate. Default: 25",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional output width. Height keeps source aspect ratio.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=3,
        help="JPEG quality for ffmpeg -q:v. Lower is better. Default: 3",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before writing frames.",
    )
    parser.add_argument(
        "--kmodel",
        default="best_fp16.kmodel",
        help="Kmodel name used when printing the board command.",
    )
    parser.add_argument(
        "--debug",
        default="0",
        help="Debug mode used when printing the board command.",
    )
    return parser.parse_args()


def require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit(
            "ffmpeg was not found. Install ffmpeg first, then rerun this tool."
        )
    return ffmpeg


def build_filter(fps: float, width: int | None) -> str:
    filters = [f"fps={fps:g}"]
    if width:
        filters.append(f"scale={width}:-2")
    return ",".join(filters)


def main() -> int:
    args = parse_args()
    ffmpeg = require_ffmpeg()

    video = Path(args.video)
    if not video.is_file():
        raise SystemExit(f"input video not found: {video}")

    out_dir = Path(args.out) if args.out else Path(f"{video.stem}_frames")
    if out_dir.exists() and args.clean:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(out_dir / "%06d.jpg")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video),
        "-vf",
        build_filter(args.fps, args.width),
        "-q:v",
        str(args.quality),
        pattern,
    ]

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    frame_count = len(list(out_dir.glob("*.jpg")))
    if frame_count == 0:
        raise SystemExit(f"no frames generated in: {out_dir}")

    print()
    print(f"Generated {frame_count} frames in: {out_dir}")
    print("Copy this directory to the board, then run:")
    print(f"./falldown_detect.elf {args.kmodel} {out_dir.name} {args.debug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
