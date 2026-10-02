#!/usr/bin/env python3
"""Profile the Q8 decode kernel with Nsight Compute.

Use this after a Q8 Linear kernel change. The tool profiles only the named
decode kernel so memory throughput, occupancy, and warp stall metrics are not
obscured by model-load work or unrelated kernels.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build-cuda/mini-llama")
    parser.add_argument(
        "--model", default="models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument(
        "--launch-skip",
        type=int,
        default=0,
        help="Number of matching Q80 decode kernel launches to skip",
    )
    parser.add_argument(
        "--launch-count",
        type=int,
        default=1,
        help="Number of matching Q80 decode kernel launches to profile",
    )
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--out", default="docs/profiles/rtx4090/q80-decode")
    args = parser.parse_args()
    binary = (ROOT / args.binary).resolve()
    out = (ROOT / args.out).resolve()
    if not binary.exists():
        parser.error(f"binary not found: {binary}")
    if shutil.which(args.ncu) is None:
        parser.error(f"ncu not found: {args.ncu}")
    if args.tokens <= 0 or args.launch_skip < 0 or args.launch_count <= 0:
        parser.error("token and launch counts must be positive; skip must be non-negative")

    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.ncu,
        "--force-overwrite",
        "--set",
        "full",
        "--kernel-name",
        "regex:Q80LinearDecodeWarpKernel",
        "--launch-skip",
        str(args.launch_skip),
        "--launch-count",
        str(args.launch_count),
        "--export",
        str(out),
        str(binary),
        "bench",
        args.model,
        "--prompt",
        "hello",
        "--n-predict",
        str(args.tokens),
        "--seed",
        str(args.seed),
        "--backend",
        "cuda",
        "--device",
        str(args.device),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(f"Wrote {out}.ncu-rep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
