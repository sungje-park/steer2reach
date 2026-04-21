"""
Launch PubSub training runs across multiple input dimensions.

Example:
    python examples/pubsub_dim_sweep.py --dims 6,10,20,40 --losses vipinns,fspinns -- -i 50000 -tc 1 -w 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PUBSUB_SCRIPT = SCRIPT_DIR / "pubsub_example.py"


def parse_int_csv(raw: str, name: str) -> List[int]:
    values = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError(f"--{name} must include at least one value")
    return values


def parse_str_csv(raw: str, name: str) -> List[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"--{name} must include at least one value")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PubSub dimension sweeps by launching examples/pubsub_example.py repeatedly."
    )
    parser.add_argument(
        "--dims",
        type=str,
        default="6,10,20,40",
        help="Comma-separated total input dimensions (d_in). Default: 6,10,20,40",
    )
    parser.add_argument(
        "--losses",
        type=str,
        default="vipinns",
        help="Comma-separated loss methods. Default: vipinns",
    )
    parser.add_argument(
        "--base_tag",
        type=str,
        default="pubsub_dim_sweep",
        help="Base run tag prefix added when no --tag is already passed through.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use for child runs.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing.",
    )
    return parser


def has_tag_or_name_passthrough(extra_args: List[str], short: str, long: str) -> bool:
    return short in extra_args or long in extra_args


def main() -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    dims = parse_int_csv(args.dims, "dims")
    losses = parse_str_csv(args.losses, "losses")

    for d_in in dims:
        if d_in < 2:
            raise ValueError(f"Each d_in must be >= 2, got {d_in}")

    set_tag = not has_tag_or_name_passthrough(extra, "-tag", "--tag")
    set_run_name = not has_tag_or_name_passthrough(extra, "-rn", "--run_name")

    total = len(dims) * len(losses)
    print(f"Launching {total} PubSub runs")
    print(f"  dims   : {dims}")
    print(f"  losses : {losses}")
    print(f"  script : {PUBSUB_SCRIPT}")

    failures = []
    for d_in in dims:
        for loss in losses:
            cmd = [args.python, str(PUBSUB_SCRIPT), "-l", loss, "--d_in", str(d_in)]
            if set_tag:
                cmd += ["-tag", f"{args.base_tag}_d{d_in}_{loss}"]
            if set_run_name:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cmd += ["-rn", f"pubsub{d_in}d_{loss}_{ts}"]
            cmd += extra

            print(" ".join(cmd))
            if args.dry_run:
                continue

            proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
            if proc.returncode != 0:
                failures.append((d_in, loss, proc.returncode))

    if failures:
        print("\nFailures:")
        for d_in, loss, code in failures:
            print(f"  d_in={d_in}, loss={loss}, returncode={code}")
        return 1

    print("\nSweep complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
