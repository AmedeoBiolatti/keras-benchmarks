#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import wandb

MS_PER_STEP_REGEX = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*ms/step")


@dataclass
class RunSpec:
    label: str  # baseline | branch
    python_exe: str
    bench_script: str
    rep: int
    steps_per_execution: int


@dataclass
class RunResult:
    spec: RunSpec
    ok: bool
    wall_s: float
    ms_per_step: float | None
    stdout: str
    stderr: str
    returncode: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-python", required=True)
    parser.add_argument("--branch-python", required=True)
    parser.add_argument("--bench-script", required=True)
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Root of the keras-benchmarks repo. Subprocesses run with this cwd.",
    )
    parser.add_argument("--output-dir", default="bench_outputs")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--steps-per-execution", default="1,8,32,128")
    parser.add_argument("--backend", default="jax")
    parser.add_argument("--timeout-s", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")

    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", default="")

    return parser.parse_args()


def ensure_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.is_dir():
        raise IsADirectoryError(path)


def ensure_dir(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if not p.is_dir():
        raise NotADirectoryError(path)


def script_path_to_module(script_path: str, repo_root: str) -> str:
    script = Path(script_path).resolve()
    root = Path(repo_root).resolve()
    try:
        rel = script.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"bench script {script} is not under repo root {root}"
        ) from exc
    return ".".join(rel.with_suffix("").parts)


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    idx = int(round((len(ys) - 1) * q))
    idx = max(0, min(idx, len(ys) - 1))
    return ys[idx]


def summarize(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {
            "n": 0.0,
            "median": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
        }
    return {
        "n": float(len(xs)),
        "median": quantile(xs, 0.50),
        "p90": quantile(xs, 0.90),
        "min": min(xs),
        "max": max(xs),
        "mean": sum(xs) / len(xs),
    }


def parse_ms_per_step(stdout: str) -> float:
    matches = MS_PER_STEP_REGEX.findall(stdout)
    if not matches:
        raise ValueError("Could not find 'ms/step' in stdout.")
    return float(matches[-1])


def make_env(backend: str, steps_per_execution: int, repo_root: str) -> dict[str, str]:
    env = os.environ.copy()
    env["KERAS_BACKEND"] = backend
    env["KERAS_STEPS_PER_EXECUTION"] = str(steps_per_execution)

    # Make import resolution robust even if a script is launched incorrectly.
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_root_abs = str(Path(repo_root).resolve())
    env["PYTHONPATH"] = (
        repo_root_abs if not existing_pythonpath else f"{repo_root_abs}{os.pathsep}{existing_pythonpath}"
    )

    # reduce CPU-side noise
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    return env


def run_one(
    spec: RunSpec,
    backend: str,
    repo_root: str,
    output_dir: Path,
    timeout_s: int | None,
) -> RunResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    bench_out = output_dir / (
        f"{spec.label}_spe{spec.steps_per_execution}_rep{spec.rep}.txt"
    )

    bench_module = script_path_to_module(spec.bench_script, repo_root)
    cmd = [
        spec.python_exe,
        "-m",
        bench_module,
        str(bench_out),
    ]

    env = make_env(
        backend=backend,
        steps_per_execution=spec.steps_per_execution,
        repo_root=repo_root,
    )

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=timeout_s,
        check=False,
        cwd=repo_root,
    )
    t1 = time.perf_counter()

    stem = f"{spec.label}_spe{spec.steps_per_execution}_rep{spec.rep}"
    (log_dir / f"{stem}.stdout").write_text(proc.stdout, encoding="utf-8")
    (log_dir / f"{stem}.stderr").write_text(proc.stderr, encoding="utf-8")

    ms_per_step = None
    ok = proc.returncode == 0
    if ok:
        try:
            ms_per_step = parse_ms_per_step(proc.stdout)
        except Exception:
            ok = False

    return RunResult(
        spec=spec,
        ok=ok,
        wall_s=t1 - t0,
        ms_per_step=ms_per_step,
        stdout=proc.stdout,
        stderr=proc.stderr,
        returncode=proc.returncode,
    )


def save_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()

    ensure_file(args.baseline_python)
    ensure_file(args.branch_python)
    ensure_file(args.bench_script)
    ensure_dir(args.repo_root)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "runs.jsonl"

    spe_values = [int(x.strip()) for x in args.steps_per_execution.split(",") if x.strip()]
    if not spe_values:
        raise ValueError("No valid steps_per_execution values provided.")

    timeout_s = None if args.timeout_s <= 0 else args.timeout_s

    specs: list[RunSpec] = []
    for spe in spe_values:
        for rep in range(args.reps):
            specs.append(
                RunSpec(
                    label="baseline",
                    python_exe=args.baseline_python,
                    bench_script=args.bench_script,
                    rep=rep,
                    steps_per_execution=spe,
                )
            )
            specs.append(
                RunSpec(
                    label="branch",
                    python_exe=args.branch_python,
                    bench_script=args.bench_script,
                    rep=rep,
                    steps_per_execution=spe,
                )
            )

    if args.shuffle:
        random.shuffle(specs)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=[x for x in args.wandb_tags.split(",") if x],
        config={
            "baseline_python": args.baseline_python,
            "branch_python": args.branch_python,
            "bench_script": args.bench_script,
            "repo_root": str(Path(args.repo_root).resolve()),
            "backend": args.backend,
            "reps": args.reps,
            "steps_per_execution": spe_values,
            "shuffle": args.shuffle,
        },
    )

    all_results: list[RunResult] = []

    for spec in specs:
        result = run_one(
            spec=spec,
            backend=args.backend,
            repo_root=args.repo_root,
            output_dir=output_dir,
            timeout_s=timeout_s,
        )
        all_results.append(result)

        record = {
            "spec": asdict(spec),
            "ok": result.ok,
            "wall_s": result.wall_s,
            "ms_per_step": result.ms_per_step,
            "returncode": result.returncode,
            "stdout_log": str(output_dir / "logs" / f"{spec.label}_spe{spec.steps_per_execution}_rep{spec.rep}.stdout"),
            "stderr_log": str(output_dir / "logs" / f"{spec.label}_spe{spec.steps_per_execution}_rep{spec.rep}.stderr"),
            "stdout_tail": "\n".join(result.stdout.splitlines()[-50:]),
            "stderr_tail": "\n".join(result.stderr.splitlines()[-100:]),
        }
        save_jsonl(jsonl_path, record)

        wandb.log(
            {
                "label": spec.label,
                "rep": spec.rep,
                "steps_per_execution": spec.steps_per_execution,
                "ok": int(result.ok),
                "wall_s": result.wall_s,
                "ms_per_step": result.ms_per_step if result.ms_per_step is not None else math.nan,
            }
        )

    summary_payload: dict[str, Any] = {}

    for spe in spe_values:
        baseline_vals = [
            r.ms_per_step
            for r in all_results
            if r.ok and r.spec.label == "baseline" and r.spec.steps_per_execution == spe and r.ms_per_step is not None
        ]
        branch_vals = [
            r.ms_per_step
            for r in all_results
            if r.ok and r.spec.label == "branch" and r.spec.steps_per_execution == spe and r.ms_per_step is not None
        ]

        baseline_stats = summarize(baseline_vals)
        branch_stats = summarize(branch_vals)

        for k, v in baseline_stats.items():
            summary_payload[f"baseline/spe_{spe}/{k}"] = v
        for k, v in branch_stats.items():
            summary_payload[f"branch/spe_{spe}/{k}"] = v

        if baseline_stats["n"] > 0 and branch_stats["n"] > 0:
            summary_payload[f"speedup/spe_{spe}/median"] = baseline_stats["median"] / branch_stats["median"]
            summary_payload[f"speedup/spe_{spe}/mean"] = baseline_stats["mean"] / branch_stats["mean"]

    wandb.log(summary_payload)

    run.summary["jsonl_path"] = str(jsonl_path)
    run.summary["logs_dir"] = str(output_dir / "logs")
    run.summary["num_runs"] = len(all_results)
    run.summary["num_success"] = sum(int(r.ok) for r in all_results)
    run.finish()

    print(
        json.dumps(
            {
                "jsonl_path": str(jsonl_path),
                "logs_dir": str(output_dir / "logs"),
                "num_runs": len(all_results),
                "num_success": sum(int(r.ok) for r in all_results),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
