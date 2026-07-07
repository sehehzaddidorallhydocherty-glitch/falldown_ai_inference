#!/usr/bin/env python3
"""Pull K230 fall evidence events and build MP4 clips automatically.

This host-side helper watches the board evidence directory over adb, pulls each
new completed event directory, then runs ffmpeg over frames/%06d.jpg.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_REMOTE_ROOT = "/sharefs/sdcard/evidence"
DEFAULT_LOCAL_ROOT = (
    "/home/ubuntu/k230_sdk/src/big/mpp/userapps/sample/"
    "falldown_ai_inference_0606/data/evidence"
)
DEFAULT_FPS = 8
EVENT_NAME_RE = re.compile(r"^[0-9]{8}_[0-9]{6}(?:_[0-9]+)?$")
EVENT_NAME_SCAN_RE = re.compile(r"(?<![0-9_])[0-9]{8}_[0-9]{6}(?:_[0-9]+)?(?![0-9_])")
IMAGE_NAME_SCAN_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_.-])(?:snapshot|\d{6})\.jpe?g(?![A-Za-z0-9_.-])"
)
ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
IMAGE_EXTS = {".jpg", ".jpeg"}


class CommandError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_command(args: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            args,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise CommandError(f"command not found: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CommandError(f"command timed out: {' '.join(args)}") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise CommandError(f"{' '.join(args)} failed: {detail}")
    return result


def clean_adb_lines(text: str) -> List[str]:
    text = normalize_adb_text(text)
    lines: List[str] = []
    for raw_line in text.replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line or line in {".", ".."}:
            continue
        if "No such file" in line or "not found" in line.lower():
            continue
        lines.extend(part for part in line.split() if part not in {".", ".."})
    return lines


def normalize_adb_text(text: str) -> str:
    text = ANSI_ESCAPE_RE.sub("", text)
    text = CONTROL_CHAR_RE.sub("", text)
    return text


def adb_shell_ls(adb: str, remote_path: str, debug: bool = False) -> List[str]:
    result = run_command([adb, "shell", "ls", remote_path], timeout=30)
    combined = f"{result.stdout}\n{result.stderr}"
    names = clean_adb_lines(combined)
    if debug:
        print(f"[{now_iso()}] adb ls {remote_path} stdout:", flush=True)
        print(result.stdout.rstrip() or "<empty>", flush=True)
        print(f"[{now_iso()}] adb ls {remote_path} stderr:", flush=True)
        print(result.stderr.rstrip() or "<empty>", flush=True)
        print(f"[{now_iso()}] adb ls {remote_path} parsed:", flush=True)
        print(" ".join(names) or "<empty>", flush=True)
    return names


def list_remote_events(adb: str, remote_root: str, debug: bool = False) -> List[str]:
    names = adb_shell_ls(adb, remote_root, debug=debug)
    events = EVENT_NAME_SCAN_RE.findall(" ".join(names))
    return sorted(set(events))


def selected_remote_events(args: argparse.Namespace, event_names: List[str]) -> List[str]:
    if args.event:
        return [args.event]
    ignored = getattr(args, "_ignored_remote_events", set())
    event_names = [name for name in event_names if name not in ignored]
    if args.process_all_remote:
        return event_names
    return event_names[-1:]


def remote_scan_mode(args: argparse.Namespace) -> str:
    if args.event:
        return f"event={args.event}"
    if args.process_all_remote:
        return "all"
    return "latest"


def remote_event_info(adb: str, remote_root: str, event_name: str, debug: bool = False) -> Optional[dict]:
    event_path = f"{remote_root.rstrip('/')}/{event_name}"
    try:
        event_children = adb_shell_ls(adb, event_path, debug=debug)
    except CommandError:
        return None

    event_child_names = IMAGE_NAME_SCAN_RE.findall(" ".join(event_children))
    has_snapshot = any(name.lower() in {"snapshot.jpg", "snapshot.jpeg"} for name in event_child_names)

    try:
        frame_names = adb_shell_ls(adb, f"{event_path}/frames", debug=debug)
    except CommandError:
        frame_names = []

    frame_image_names = IMAGE_NAME_SCAN_RE.findall(" ".join(frame_names))
    frame_count = sum(1 for name in frame_image_names if re.fullmatch(r"(?i)[0-9]{6}\.jpe?g", name))
    return {
        "event_name": event_name,
        "remote_path": event_path,
        "has_snapshot": has_snapshot,
        "frame_count": frame_count,
    }


def read_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"processed": {}}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"processed": {}}


def write_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(state_path)


def local_frame_count(event_dir: Path) -> int:
    frames_dir = event_dir / "frames"
    if not frames_dir.is_dir():
        return 0
    return sum(1 for path in frames_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def pull_event(adb: str, remote_path: str, local_root: Path) -> Path:
    local_root.mkdir(parents=True, exist_ok=True)
    run_command([adb, "pull", remote_path, str(local_root)], timeout=300)
    return local_root / Path(remote_path).name


def remove_local_event_for_resync(event_dir: Path, local_root: Path) -> None:
    if not event_dir.exists():
        return

    resolved_event = event_dir.resolve()
    resolved_root = local_root.resolve()
    if resolved_event == resolved_root or resolved_root not in resolved_event.parents:
        raise CommandError(f"refuse to remove unsafe local event path: {event_dir}")

    print(f"[{now_iso()}] resync removing stale local event: {event_dir}", flush=True)
    shutil.rmtree(event_dir)


def build_video(ffmpeg: str, event_dir: Path, fps: int, overwrite: bool) -> Path:
    frames_pattern = event_dir / "frames" / "%06d.jpg"
    video_path = event_dir / "event.mp4"
    if video_path.exists() and not overwrite:
        return video_path

    args = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-framerate",
        str(fps),
        "-i",
        str(frames_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    run_command(args, timeout=300)
    return video_path


def write_event_metadata(event_dir: Path, info: dict, fps: int, video_path: Path) -> None:
    metadata = {
        "event_name": info["event_name"],
        "remote_path": info["remote_path"],
        "pulled_at": now_iso(),
        "fps": fps,
        "frame_count": local_frame_count(event_dir),
        "snapshot": str(event_dir / "snapshot.jpg"),
        "video": str(video_path),
        "status": "processed",
    }
    (event_dir / "event.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def rebuild_events_index(local_root: Path, index_path: Path) -> None:
    events = []
    if local_root.is_dir():
        for event_dir in sorted(local_root.iterdir()):
            metadata_path = event_dir / "event.json"
            if not event_dir.is_dir() or not metadata_path.exists():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            events.append(metadata)

    index = {
        "updated_at": now_iso(),
        "event_count": len(events),
        "events": sorted(events, key=lambda item: item.get("event_name", ""), reverse=True),
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(index_path)


def should_skip_event(state: dict, event_name: str, event_dir: Path, overwrite: bool) -> bool:
    if overwrite:
        return False
    processed = state.get("processed", {})
    if event_name in processed and (event_dir / "event.mp4").exists():
        return True
    return False


def finish_local_event(
    args: argparse.Namespace,
    state: dict,
    info: dict,
    event_dir: Path,
    local_root: Path,
    source: str,
) -> None:
    event_name = info["event_name"]
    frame_count = local_frame_count(event_dir)
    if frame_count <= 0:
        print(f"[{now_iso()}] skip {event_name}: no local frames", flush=True)
        return

    print(
        f"[{now_iso()}] building mp4 for {event_name}, "
        f"source={source}, frames={frame_count}, fps={args.fps}",
        flush=True,
    )
    video_path = build_video(args.ffmpeg, event_dir, args.fps, args.overwrite)
    write_event_metadata(event_dir, info, args.fps, video_path)

    state.setdefault("processed", {})[event_name] = {
        "processed_at": now_iso(),
        "frame_count": frame_count,
        "video": str(video_path),
    }
    write_state(Path(args.state_file).expanduser(), state)
    rebuild_events_index(local_root, Path(args.index_file).expanduser())
    print(f"[{now_iso()}] processed {event_name}: {video_path}", flush=True)


def process_event(args: argparse.Namespace, state: dict, info: dict) -> None:
    event_name = info["event_name"]
    local_root = Path(args.local_root).expanduser()
    event_dir = local_root / event_name

    if should_skip_event(state, event_name, event_dir, args.overwrite) and not args.resync:
        if args.verbose:
            print(f"[{now_iso()}] skip remote {event_name}: already processed locally", flush=True)
        return

    if args.resync:
        remove_local_event_for_resync(event_dir, local_root)

    print(f"[{now_iso()}] pulling {info['remote_path']} -> {local_root}", flush=True)
    event_dir = pull_event(args.adb, info["remote_path"], local_root)
    finish_local_event(args, state, info, event_dir, local_root, source="adb")


def process_existing_local_events(args: argparse.Namespace, state: dict) -> None:
    local_root = Path(args.local_root).expanduser()
    if not local_root.is_dir():
        return

    for event_dir in sorted(local_root.iterdir()):
        event_name = event_dir.name
        if not event_dir.is_dir() or not EVENT_NAME_RE.match(event_name):
            continue
        if should_skip_event(state, event_name, event_dir, args.overwrite):
            continue
        if not (event_dir / "snapshot.jpg").exists():
            continue
        if local_frame_count(event_dir) < args.min_frames:
            continue

        info = {
            "event_name": event_name,
            "remote_path": f"{args.remote_root.rstrip('/')}/{event_name}",
            "has_snapshot": True,
            "frame_count": local_frame_count(event_dir),
        }
        finish_local_event(args, state, info, event_dir, local_root, source="local")


def stable_events(
    args: argparse.Namespace,
    observations: Dict[str, dict],
) -> Iterable[dict]:
    event_names = list_remote_events(args.adb, args.remote_root, debug=args.debug_adb)
    target_event_names = selected_remote_events(args, event_names)
    now = time.monotonic()
    last_status = getattr(args, "_last_status", 0.0)
    if last_status == 0.0 or now - last_status >= args.status_seconds:
        targets = ", ".join(target_event_names) if target_event_names else "-"
        print(
            f"[{now_iso()}] scan: remote_events={len(event_names)}, "
            f"mode={remote_scan_mode(args)}, selected={len(target_event_names)}, "
            f"target={targets}",
            flush=True,
        )
        if args.verbose and event_names:
            print(f"[{now_iso()}] remote event names: {', '.join(event_names)}", flush=True)
        if args.event and args.event not in event_names:
            print(
                f"[{now_iso()}] waiting for requested remote event: {args.event}",
                flush=True,
            )
        setattr(args, "_last_status", now)

    for event_name in target_event_names:
        info = remote_event_info(args.adb, args.remote_root, event_name, debug=args.debug_adb)
        if not info:
            continue

        frame_count = info["frame_count"]
        ready = info["has_snapshot"] and frame_count >= args.min_frames
        obs = observations.setdefault(
            event_name,
            {
                "last_frame_count": -1,
                "stable_since": now,
                "reported_wait": False,
            },
        )

        if not ready:
            if not obs["reported_wait"]:
                print(
                    f"[{now_iso()}] waiting {event_name}: "
                    f"snapshot={int(info['has_snapshot'])}, frames={frame_count}",
                    flush=True,
                )
                obs["reported_wait"] = True
            obs["last_frame_count"] = frame_count
            obs["stable_since"] = now
            continue

        if args.once:
            print(f"[{now_iso()}] ready {event_name}: frames={frame_count}", flush=True)
            yield info
            continue

        if frame_count != obs["last_frame_count"]:
            obs["last_frame_count"] = frame_count
            obs["stable_since"] = now
            obs["reported_wait"] = False
            print(f"[{now_iso()}] observing {event_name}: frames={frame_count}", flush=True)
            continue

        stable_for = now - obs["stable_since"]
        if stable_for >= args.stable_seconds:
            yield info


def monitor(args: argparse.Namespace) -> int:
    local_root = Path(args.local_root).expanduser()
    local_root.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_file).expanduser()
    state = read_state(state_path)
    observations: Dict[str, dict] = {}
    rebuild_events_index(local_root, Path(args.index_file).expanduser())
    if args.ignore_existing and not args.event:
        ignored_events = set(list_remote_events(args.adb, args.remote_root, debug=args.debug_adb))
        setattr(args, "_ignored_remote_events", ignored_events)
    else:
        ignored_events = set()
        setattr(args, "_ignored_remote_events", ignored_events)

    print(f"[{now_iso()}] remote evidence root: {args.remote_root}", flush=True)
    print(f"[{now_iso()}] local evidence root:  {local_root}", flush=True)
    print(f"[{now_iso()}] ffmpeg fps:           {args.fps}", flush=True)
    if ignored_events:
        print(
            f"[{now_iso()}] ignoring existing remote events: {len(ignored_events)}",
            flush=True,
        )
    if args.process_local_existing:
        process_existing_local_events(args, state)

    while True:
        try:
            if args.process_local_existing:
                process_existing_local_events(args, state)
            for info in stable_events(args, observations):
                process_event(args, state, info)
        except CommandError as exc:
            print(f"[{now_iso()}] warning: {exc}", file=sys.stderr, flush=True)

        if args.once:
            break
        time.sleep(args.poll_seconds)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically adb-pull K230 fall evidence and build event.mp4 files.",
    )
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--local-root", default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--stable-seconds", type=float, default=8.0)
    parser.add_argument("--status-seconds", type=float, default=10.0)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--state-file", default=str(Path(DEFAULT_LOCAL_ROOT) / ".auto_evidence_state.json"))
    parser.add_argument("--index-file", default=str(Path(DEFAULT_LOCAL_ROOT) / "events_index.json"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resync", action="store_true", help="re-pull remote events even if local outputs already exist")
    parser.add_argument("--all", dest="process_all_remote", action="store_true", help="process every remote event folder instead of only the newest one")
    parser.add_argument("--event", help="process one named remote event folder, for example 19700101_001429")
    parser.add_argument("--ignore-existing", action="store_true", help="ignore board event folders that already exist when the service starts")
    parser.add_argument("--process-local-existing", action="store_true", help="also build videos for existing local event folders")
    parser.add_argument("--debug-adb", action="store_true", help="print raw adb shell ls stdout/stderr while scanning")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--once", action="store_true", help="scan once, then exit")
    args = parser.parse_args()
    if args.event and args.process_all_remote:
        parser.error("--event cannot be used together with --all")
    if args.event and not EVENT_NAME_RE.match(args.event):
        parser.error("--event must look like YYYYMMDD_HHMMSS, for example 19700101_001429")
    return args


if __name__ == "__main__":
    sys.exit(monitor(parse_args()))
