#!/usr/bin/env python3
"""Collect reproducible Nsight Systems profiles for mini-llama CUDA paths.

The script profiles Prefill and Decode as separate processes so their kernel
summaries are directly comparable. It does not alter inference behavior.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = "cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum"


def run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def profile_case(args: argparse.Namespace, name: str,
                 bench_args: list[str]) -> dict[str, str]:
    prefix = args.out_dir / name
    nsys_command = [
        args.nsys,
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,osrt",
        "--sample=none",
        "--output",
        str(prefix),
        *bench_args,
    ]
    profile = run(nsys_command, args.timeout)
    report_path = prefix.with_suffix(".nsys-rep")
    if not report_path.exists():
        raise RuntimeError(f"Nsight report missing: {report_path}")

    stats_command = [
        args.nsys,
        "stats",
        "--force-export=true",
        "--report",
        REPORTS,
        str(report_path),
    ]
    stats = run(stats_command, args.timeout)
    stats_path = prefix.with_name(f"{prefix.name}-summary.txt")
    stats_path.write_text(stats.stdout + stats.stderr, encoding="utf-8")

    benchmark_path = prefix.with_name(f"{prefix.name}-benchmark.txt")
    benchmark_path.write_text(profile.stdout + profile.stderr, encoding="utf-8")
    return {
        "name": name,
        "report": str(report_path.relative_to(ROOT)),
        "summary": str(stats_path.relative_to(ROOT)),
        "benchmark_log": str(benchmark_path.relative_to(ROOT)),
        "command": " ".join(nsys_command),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build-cuda/mini-llama")
    parser.add_argument(
        "--model", default="models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefill-tokens", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument(
        "--q8-prefill",
        choices=("q8", "f32", "f16", "auto"),
        default="q8",
        help="Q8 Prefill profile for the Prefill trace",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--out-dir", default="docs/profiles/rtx4090")
    args = parser.parse_args()
    args.binary = (ROOT / args.binary).resolve()
    args.out_dir = (ROOT / args.out_dir).resolve()
    if not args.binary.exists():
        parser.error(f"binary not found: {args.binary}")
    if shutil.which(args.nsys) is None:
        parser.error(f"nsys not found: {args.nsys}")
    if args.prefill_tokens <= 0 or args.decode_tokens <= 0:
        parser.error("token counts must be positive")
    return args


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    common = [
        str(args.binary),
        "bench",
        args.model,
        "--seed",
        str(args.seed),
        "--backend",
        "cuda",
        "--device",
        str(args.device),
    ]
    cases = [
        profile_case(
            args,
            "prefill",
            [
                *common,
                "--prompt-tokens",
                str(args.prefill_tokens),
                "--n-predict",
                "1",
                "--q8-prefill",
                args.q8_prefill,
            ],
        ),
        profile_case(
            args,
            "decode",
            [
                *common,
                "--prompt",
                "hello",
                "--n-predict",
                str(args.decode_tokens),
            ],
        ),
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "device": args.device,
        "seed": args.seed,
        "prefill_tokens": args.prefill_tokens,
        "decode_tokens": args.decode_tokens,
        "q8_prefill": args.q8_prefill,
        "reports": cases,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {manifest_path}")
    for case in cases:
        print(f"Wrote {case['report']}")
        print(f"Wrote {case['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
