#!/usr/bin/env python3
"""Measure reproducible long-context Prefill performance.

The benchmark binary's --prompt-tokens option feeds a repeated BOS token. This
keeps token count exact and measures runtime behavior without tying results to
tokenizer segmentation or prompt text.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_prefill_ms(stdout: str) -> float:
    match = re.search(r"prefill time:\s*([0-9.]+) ms", stdout)
    if not match:
        raise RuntimeError(f"prefill time missing from output:\n{stdout}")
    return float(match.group(1))


def run_case(args: argparse.Namespace, prompt_tokens: int) -> dict:
    command = [
        str(args.binary),
        "bench",
        args.model,
        "--prompt-tokens",
        str(prompt_tokens),
        "--n-predict",
        "1",
        "--seed",
        str(args.seed),
        "--backend",
        args.backend,
    ]
    if args.backend == "cuda":
        command.extend(["--device", str(args.device)])
        if args.q8_prefill != "q8":
            command.extend(["--q8-prefill", args.q8_prefill])
    if args.backend == "cpu":
        command.extend(["--threads", str(args.threads)])

    timings = []
    outputs = []
    for _ in range(args.runs):
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"command failed: {' '.join(command)}\n{result.stdout}\n"
                f"{result.stderr}"
            )
        timings.append(parse_prefill_ms(result.stdout))
        outputs.append(result.stdout)

    median_ms = statistics.median(timings)
    return {
        "prompt_tokens": prompt_tokens,
        "runs": timings,
        "median_prefill_ms": median_ms,
        "median_prefill_tokens_per_sec": prompt_tokens * 1000.0 / median_ms,
        "command": command,
        "last_output": outputs[-1],
    }


def render_markdown(args: argparse.Namespace, results: list[dict]) -> str:
    lines = [
        f"# Prefill Profile — {datetime.now(timezone.utc).date().isoformat()}",
        "",
        "Synthetic repeated-BOS prompts provide exact token counts; results "
        "measure Prefill rather than text quality.",
        "",
        f"- Model: `{args.model}`",
        f"- Backend: `{args.backend}`",
        f"- Q8 Prefill profile: `{args.q8_prefill}`",
        f"- Runs per length: {args.runs}; median reported",
        "",
        "| Prompt tokens | Prefill runs (ms) | Median Prefill (ms) | Prefill tok/s |",
        "|---:|---|---:|---:|",
    ]
    for result in results:
        runs = " / ".join(f"{value:.2f}" for value in result["runs"])
        lines.append(
            f"| {result['prompt_tokens']} | {runs} | "
            f"{result['median_prefill_ms']:.2f} | "
            f"{result['median_prefill_tokens_per_sec']:.2f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Compare runs only when the model, backend, seed, prompt lengths, "
        "repetition count, and benchmark flags are identical. Use the emitted "
        "command and the Q8 Prefill profile field to distinguish the default "
        "quantized path from optional VRAM-for-throughput experiments.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build-cuda/mini-llama")
    parser.add_argument(
        "--model", default="models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf"
    )
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--q8-prefill",
        choices=("q8", "f32", "f16", "auto"),
        default="q8",
        help="Q8 Prefill profile; auto selects F32 cache within VRAM budget",
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--lengths", nargs="+", type=int,
                        default=[1, 32, 256, 1024])
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--out", default="docs/benchmarks/prefill-profile.md")
    args = parser.parse_args()
    args.binary = (ROOT / args.binary).resolve()
    args.out = (ROOT / args.out).resolve()
    if args.runs <= 0 or any(length <= 0 for length in args.lengths):
        parser.error("--runs and all --lengths values must be positive")
    if args.q8_prefill != "q8" and args.backend != "cuda":
        parser.error("Q8 Prefill cache options require --backend cuda")
    if not args.binary.exists():
        parser.error(f"binary not found: {args.binary}")
    return args


def main() -> int:
    args = parse_args()
    results = [run_case(args, length) for length in args.lengths]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(args, results), encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
