# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the explicit Figure 3 configurations from the layered-prefill paper."""

import argparse
import contextlib
import copy
import csv
import hashlib
import importlib.util
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "benchmarks/configs/layered_prefill/figure3.json"
CLI = [sys.executable, "-m", "vllm.entrypoints.cli.main"]
ENV = {
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
}
NOTES = [
    "Figure 3 uses H100 80GB x2 with NVLink. Other GPUs are hardware overrides.",
    "Both modes use Model Runner V2, synchronous scheduling, no prefix cache.",
    (
        "Both modes use breakable prefill CUDA graphs and full decode CUDA graphs, "
        "without torch.compile. --enforce-eager disables graphs for both. Capture "
        "sizes are powers of two through each mode's token budget, plus the budget."
    ),
    (
        "Fixed num_layer_groups PER PP RANK (Qwen 16, GPT-OSS 12) comes from the "
        "reference code. The paper adapts groups to input length and advances "
        "decode every iteration; this implementation drains a fixed admitted batch. "
        "Matching Figure 3 workloads does not reproduce that scheduling policy."
    ),
    (
        "Layered execution keeps all requests selected within max_num_batched_tokens "
        "in one batch across layer groups. PP does not divide this token budget."
    ),
    (
        "Token chunking stays enabled in BOTH modes: the original layered scheduler "
        "also chunks inputs at its 8192-token budget, below its 32768 context limit."
    ),
    (
        "Serving imports the original dataset sampler; vLLM's default ShareGPT "
        "sampler has different prompt lengths. Energy/Zeus measurements are not ported."
    ),
    (
        "arXiv requests exceeding max_model_len (prompt plus requested output) are "
        "skipped in source order and replaced with subsequent articles, without "
        "truncation. Paired schedulers share the same filtered requests."
    ),
    "Latency uses synthetic lengths; Figure 3 SLO curves require --benchmark serve.",
    "GPT-OSS uses a BF16-converted checkpoint to match the paper's weight precision.",
]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_config(config: dict) -> list[dict]:
    """Expand each panel into paired schedules at the published request rates."""
    defaults = config["benchmark_defaults"]
    rows = []
    for panel in config["panels"]:
        model = config["models"][panel["model"]]
        for schedule, settings in config["schedules"].items():
            for rate in panel["request_rates"]:
                if rate <= 0:
                    raise ValueError("Figure 3 request rates must be positive")
                rows.append(
                    {
                        **config["engine_defaults"],
                        **settings,
                        "panel": panel["id"],
                        "model_key": panel["model"],
                        "model_name": model["model_name"],
                        "schedule_mode": schedule,
                        "num_layer_groups": model["num_layer_groups"]
                        if schedule == "layered-prefill"
                        else 1,
                        "dataset_name": panel["dataset"],
                        "request_rate": rate,
                        "burstiness": defaults["burstiness"],
                        "num_requests": min(
                            defaults["num_requests"],
                            int(defaults["max_seconds"] * rate),
                        ),
                        "ttft_slo_ms": panel["ttft_slo_ms"],
                        "tbt_slo_ms": model["tbt_slo_ms"],
                    }
                )
    return rows


def model_id(value: str) -> str:
    for part in Path(value).parts:
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    return value


def engine_config(row: dict, args: argparse.Namespace) -> dict:
    source_model = row["model_name"]
    model = source_model if Path(source_model).is_dir() else model_id(source_model)
    config = {
        "model": args.model or model,
        "dtype": row["dtype"],
        "max_num_batched_tokens": int(row["max_num_batched_tokens"]),
        "max_num_seqs": int(row["max_num_seqs"]),
        "max_model_len": int(row["max_model_len"]),
        "gpu_memory_utilization": float(row["gpu_memory_utilization"]),
        "tensor_parallel_size": args.tp or int(row["tensor_parallel_size"]),
        "pipeline_parallel_size": args.pp or row["pipeline_parallel_size"],
        "enable_expert_parallel": args.ep or row["enable_expert_parallel"],
        "num_layer_groups": row["num_layer_groups"],
        "distributed_executor_backend": "mp",
        "enforce_eager": args.enforce_eager
        or str(row["enforce_eager"]).lower() == "true",
        "async_scheduling": False,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "seed": args.seed,
    }
    if not config["enforce_eager"]:
        budget = config["max_num_batched_tokens"]
        sizes = sorted({1 << i for i in range(budget.bit_length())} | {budget})
        config["compilation_config"] = {
            "mode": 0,
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": sizes,
        }
    parts = Path(source_model).parts
    if not args.model and model != source_model and "snapshots" in parts:
        config["revision"] = parts[parts.index("snapshots") + 1]
    return config


def cli_flags(config: dict) -> list[str]:
    flags = []
    for key, value in config.items():
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            flags.append("--" + ("" if value else "no-") + flag)
        elif isinstance(value, dict):
            flags.extend(["--" + flag, json.dumps(value)])
        else:
            flags.extend(["--" + flag, str(value)])
    return flags


def make_cases(
    rows: list[dict], args: argparse.Namespace
) -> tuple[list[dict], list[dict]]:
    cases = {}
    skipped = []
    for row in rows:
        if row["model_key"] not in args.models:
            continue
        if args.dataset and row["dataset_name"] not in args.dataset:
            continue
        engine = engine_config(row, args)
        if args.benchmark == "latency":
            workload = {
                "input_len": args.input_len,
                "output_len": args.output_len,
                "batch_size": args.batch_size,
                "num_iters": args.num_iters,
                "num_iters_warmup": args.num_iters_warmup,
                "disable_detokenize": True,
            }
            if args.input_len + args.output_len > engine["max_model_len"]:
                raise ValueError(
                    "Latency input_len + output_len exceeds source max_model_len"
                )
        else:
            workload = {
                "dataset": row["dataset_name"],
                "request_rate": float(row["request_rate"]),
                "num_prompts": int(row["num_requests"]),
                "burstiness": row["burstiness"],
                "ttft_slo_ms": row["ttft_slo_ms"],
                "tbt_slo_ms": row["tbt_slo_ms"],
            }
        key = digest({"engine": engine, "workload": workload})
        if key in cases:
            cases[key]["sources"].append(row)
        else:
            cases[key] = {
                "id": key,
                "engine": engine,
                "workload": workload,
                "sources": [row],
            }
    if not cases:
        raise ValueError("No experiments selected")
    return list(cases.values()), skipped


def dataset_file(output: Path, case: dict) -> Path:
    key = digest([case["engine"]["model"], case["workload"]["dataset"]])
    return output / "datasets" / f"{key}.jsonl"


def server_command(engine: dict, port: int) -> list[str]:
    return [
        *CLI,
        "serve",
        engine["model"],
        *cli_flags({key: value for key, value in engine.items() if key != "model"}),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def benchmark_command(
    case: dict, args: argparse.Namespace, port: int = 8000, warmup: bool = False
) -> list[str]:
    result_dir = args.output_dir / case["id"]
    if args.benchmark == "latency":
        return [
            *CLI,
            "bench",
            "latency",
            *cli_flags(case["engine"]),
            *cli_flags(case["workload"]),
            "--output-json",
            str(result_dir / "result.json"),
        ]
    workload = case["workload"]
    count = workload["num_prompts"]
    if warmup:
        count = min(count, int(args.warmup_seconds * workload["request_rate"]))
    return [
        *CLI,
        "bench",
        "serve",
        "--model",
        case["engine"]["model"],
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--backend",
        "vllm",
        "--endpoint",
        "/v1/completions",
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(dataset_file(args.output_dir, case)),
        "--disable-shuffle",
        "--skip-chat-template",
        "--no-oversample",
        "--request-rate",
        str(workload["request_rate"]),
        "--burstiness",
        str(workload["burstiness"]),
        "--num-prompts",
        str(count),
        "--num-warmups",
        "0",
        "--seed",
        str(args.seed),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "5,10,50,90,95,99,99.9,100",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        "warmup.json" if warmup else "result.json",
    ]


def sample_dataset(
    dataset, tokenizer, name: str, count: int, max_model_len: int
) -> tuple[list[dict], int]:
    """Select full requests that fit, advancing past oversized arXiv articles."""

    def candidates():
        if name != "arxiv":
            yield from dataset.sample(tokenizer, num_requests=count)
            return
        # The reference arXiv sampler always starts at data[0].
        sampler = copy.copy(dataset)
        for entry in dataset.data:
            sampler.data = [entry]
            yield from sampler.sample(tokenizer, num_requests=1)

    requests = []
    skipped = 0
    for request in candidates():
        length = len(tokenizer(request.prompt).input_ids)
        output_length = int(request.expected_output_len)
        if length + output_length > max_model_len:
            skipped += 1
            continue
        requests.append({"prompt": request.prompt, "output_tokens": output_length})
        if len(requests) == count:
            return requests, skipped
    raise ValueError(
        f"Expected {count} valid {name} requests, got {len(requests)}; "
        f"skipped {skipped} exceeding max_model_len={max_model_len} "
        "(prompt + output tokens). Source samples exhausted."
    )


def prepare_datasets(cases: list[dict], args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    filename = args.source_dir / "benchmarks/benchmark_dataset.py"
    spec = importlib.util.spec_from_file_location(
        "layered_prefill_source_dataset", filename
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load dataset sampler: {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    grouped = defaultdict(list)
    for case in cases:
        grouped[dataset_file(args.output_dir, case)].append(case)
    for path, group in grouped.items():
        if path.exists():
            continue
        case = group[0]
        name = case["workload"]["dataset"]
        dataset_cls = {
            "arxiv": module.ArxivDataset,
            "sharegpt": module.ShareGPTDataset,
        }.get(name)
        if dataset_cls is None:
            raise ValueError(f"Unsupported source dataset: {name}")
        tokenizer = AutoTokenizer.from_pretrained(
            case["engine"]["model"], revision=case["engine"].get("revision")
        )
        dataset = dataset_cls(random_seed=args.seed)
        count = max(c["workload"]["num_prompts"] for c in group)
        max_model_len = min(c["engine"]["max_model_len"] for c in group)
        requests, skipped = sample_dataset(
            dataset, tokenizer, name, count, max_model_len
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as file:
            for request in requests:
                file.write(json.dumps(request) + "\n")
        write_json(
            path.with_suffix(".metadata.json"),
            {
                "model": case["engine"]["model"],
                "dataset": name,
                "seed": args.seed,
                "max_model_len": max_model_len,
                "num_requests": len(requests),
                "skipped_over_context": skipped,
            },
        )
        temporary.replace(path)
        print(
            f"Prepared {len(requests)} {name} requests for {case['engine']['model']}; "
            f"skipped {skipped} over context limit {max_model_len}. Dataset: {path}",
            flush=True,
        )


def wait_for_gpus(args: argparse.Namespace) -> None:
    while True:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                f"--id={args.gpus}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        used = [float(value) for value in result.stdout.splitlines()]
        if used and all(value <= args.idle_memory_mib for value in used):
            return
        print(f"Waiting for GPUs {args.gpus}: memory used {used} MiB", flush=True)
        time.sleep(10)


@contextlib.contextmanager
def process(command: list[str], log: Path, env: dict):
    with log.open("w") as file:
        child = subprocess.Popen(
            command,
            stdout=file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=ROOT,
            start_new_session=True,
        )
        try:
            yield child
        finally:
            # The process group belongs to this invocation, including its workers.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.wait(timeout=10)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def run_command(command: list[str], log: Path, env: dict, timeout: int) -> None:
    print(shlex.join(command), flush=True)
    with process(command, log, env) as child:
        code = child.wait(timeout=timeout)
        if code:
            raise RuntimeError(f"Command exited with {code}; see {log}")


def wait_for_server(child: subprocess.Popen, port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError("Server exited during startup; see server.log")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError("Server readiness timed out; see server.log")


def is_complete(directory: Path) -> bool:
    try:
        status = json.loads((directory / "status.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        return status["status"] == "ok" and (
            "avg_latency" in result or "completed" in result
        )
    except (OSError, ValueError, KeyError):
        return False


def paper_slo_metrics(result: dict, workload: dict) -> dict:
    """Apply Table 5 to each request, including its worst inter-token latency."""
    count = workload["num_prompts"]
    ttfts, itls = result["ttfts"], result["itls"]
    errors, lengths = result["errors"], result["output_lens"]
    if not all(len(values) == count for values in (ttfts, itls, errors, lengths)):
        raise ValueError("Incomplete per-request timings for Figure 3 SLO evaluation")
    ttft_passed = tbt_passed = both_passed = 0
    for ttft, intervals, error, length in zip(ttfts, itls, errors, lengths):
        if error or length <= 0:
            continue
        if length > 1 and not intervals:
            raise ValueError("Missing inter-token timings for a multi-token response")
        meets_ttft = 0 <= ttft * 1000 <= workload["ttft_slo_ms"]
        meets_tbt = all(0 <= t * 1000 <= workload["tbt_slo_ms"] for t in intervals)
        ttft_passed += meets_ttft
        tbt_passed += meets_tbt
        both_passed += meets_ttft and meets_tbt
    return {
        "slo_attained": both_passed,
        "slo_attainment_pct": 100 * both_passed / count,
        "ttft_attainment_pct": 100 * ttft_passed / count,
        "tbt_attainment_pct": 100 * tbt_passed / count,
    }


def run_case(case: dict, args: argparse.Namespace, env: dict, port: int = 8000) -> None:
    directory = args.output_dir / case["id"]
    directory.mkdir(parents=True, exist_ok=True)
    status = directory / "status.json"
    if args.resume and is_complete(directory):
        print(f"Skipping completed {case['id']}", flush=True)
        return
    write_json(status, {"status": "running", "case": case})
    try:
        if args.benchmark == "serve" and int(
            args.warmup_seconds * case["workload"]["request_rate"]
        ):
            run_command(
                benchmark_command(case, args, port, warmup=True),
                directory / "warmup.log",
                env,
                args.timeout,
            )
        command = benchmark_command(case, args, port)
        write_json(directory / "command.json", command)
        (directory / "result.json").unlink(missing_ok=True)
        run_command(command, directory / "benchmark.log", env, args.timeout)
        result = json.loads((directory / "result.json").read_text())
        if (
            args.benchmark == "serve"
            and result["completed"] != case["workload"]["num_prompts"]
        ):
            raise RuntimeError(f"Only {result['completed']} requests completed")
        if args.benchmark == "serve":
            result.update(paper_slo_metrics(result, case["workload"]))
            write_json(directory / "result.json", result)
        write_json(status, {"status": "ok", "case": case})
    except Exception as error:
        write_json(status, {"status": "failed", "case": case, "error": str(error)})
        raise


def summarize(cases: list[dict], output: Path) -> None:
    rows = []
    for case in cases:
        directory = output / case["id"]
        status_path = directory / "status.json"
        status = (
            json.loads(status_path.read_text())
            if status_path.exists()
            else {"status": "pending"}
        )
        row = {
            "id": case["id"],
            "panels": ",".join(dict.fromkeys(s["panel"] for s in case["sources"])),
            **case["engine"],
            **case["workload"],
            "status": status["status"],
        }
        if status["status"] == "ok":
            result = json.loads((directory / "result.json").read_text())
            row.update(
                {
                    key: value
                    for key, value in result.items()
                    if key
                    in {
                        "avg_latency",
                        "completed",
                        "duration",
                        "request_throughput",
                        "output_throughput",
                        "total_token_throughput",
                        "slo_attained",
                        "slo_attainment_pct",
                        "ttft_attainment_pct",
                        "tbt_attainment_pct",
                    }
                    or key.endswith(("_ttft_ms", "_tpot_ms", "_itl_ms", "_e2el_ms"))
                }
            )
            row.update(
                {
                    f"p{key}_latency_s": value
                    for key, value in result.get("percentiles", {}).items()
                }
            )
        if "error" in status:
            row["error"] = status["error"]
        rows.append(row)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "summary.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT.parent / "layered-prefill",
        help="Reference repository, used only for serving dataset sampling",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["qwen", "gpt-oss"],
        default=["qwen", "gpt-oss"],
        help="Figure 3 models to run (both by default)",
    )
    parser.add_argument("--benchmark", choices=["serve", "latency"], default="serve")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
    parser.add_argument(
        "--model", help="Checkpoint override; requires selecting one --models entry"
    )
    parser.add_argument("--tp", type=int, help="Override original TP")
    parser.add_argument("--pp", type=int)
    parser.add_argument("--ep", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dataset", choices=["arxiv", "sharegpt"], nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-seconds", type=int)
    parser.add_argument("--relax-seconds", type=int)
    parser.add_argument("--input-len", type=int, default=8192)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-iters-warmup", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=30)
    parser.add_argument(
        "--idle-memory-mib",
        type=int,
        default=500,
        help="Wait until every selected GPU uses at most this memory",
    )
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.model and len(args.models) != 1:
        parser.error("--model requires one selected model, e.g. --models gpt-oss")
    args.config = args.config.resolve()
    args.experiment_config = json.loads(args.config.read_text())
    for key in ("warmup_seconds", "relax_seconds"):
        if getattr(args, key) is None:
            setattr(args, key, args.experiment_config["benchmark_defaults"][key])
    args.source_dir = args.source_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    for key in ("input_len", "output_len", "batch_size", "num_iters", "timeout"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    for key in ("tp", "pp"):
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            parser.error(f"--{key} must be positive")
    for key in (
        "warmup_seconds",
        "relax_seconds",
        "idle_memory_mib",
        "num_iters_warmup",
    ):
        if getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = read_config(args.experiment_config)
    cases, skipped = make_cases(rows, args)
    env = os.environ | ENV | {"CUDA_VISIBLE_DEVICES": args.gpus}
    manifest = {
        "benchmark": args.benchmark,
        "source_dir": str(args.source_dir),
        "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "experiment_config": args.experiment_config,
        "cases": cases,
        "skipped": skipped,
        "notes": NOTES,
        "environment": ENV | {"CUDA_VISIBLE_DEVICES": args.gpus},
        "seed": args.seed,
        "warmup_seconds": args.warmup_seconds,
        "relax_seconds": args.relax_seconds,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            str(file.relative_to(args.source_dir)): hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
            for file in (
                args.source_dir / "benchmarks/benchmark_dataset.py",
                *sorted(
                    (args.source_dir / "benchmarks/preprocessed_traces").glob("*.csv")
                ),
            )
            if file.exists()
        },
    }
    path = args.output_dir / "manifest.json"
    if path.exists():
        if (
            not args.resume
            and not args.dry_run
            and any(args.output_dir.glob("*/status.json"))
        ):
            raise ValueError("Output exists; use --resume or a new --output-dir")
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Existing manifest differs; choose a new --output-dir")
    write_json(path, manifest)
    print(f"Planned {len(cases)} experiments; skipped {len(skipped)}. Manifest: {path}")
    if args.dry_run:
        command_env = [
            "env",
            *(f"{key}={value}" for key, value in manifest["environment"].items()),
        ]
        for case in cases:
            if args.benchmark == "serve":
                print(shlex.join([*command_env, *server_command(case["engine"], 8000)]))
            print(shlex.join([*command_env, *benchmark_command(case, args)]))
        return
    gpu_count = len(args.gpus.split(","))
    if any(
        c["engine"]["tensor_parallel_size"] * c["engine"]["pipeline_parallel_size"]
        > gpu_count
        for c in cases
    ):
        raise ValueError(
            "Selected GPUs are fewer than TP * PP; use --gpus or override --tp/--pp"
        )
    if args.benchmark == "serve":
        prepare_datasets(cases, args)
    grouped = defaultdict(list)
    for case in cases:
        if args.resume and is_complete(args.output_dir / case["id"]):
            continue
        grouped[digest(case["engine"])].append(case)
    try:
        for engine_id, group in grouped.items():
            if args.benchmark == "latency":
                for case in group:
                    wait_for_gpus(args)
                    run_case(case, args, env)
                continue
            wait_for_gpus(args)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            command = server_command(group[0]["engine"], port)
            log = args.output_dir / f"{engine_id}.server.log"
            print(shlex.join(command), flush=True)
            with process(command, log, env) as child:
                wait_for_server(child, port, args.timeout)
                for case in group:
                    run_case(case, args, env, port)
                    summarize(cases, args.output_dir)
                    time.sleep(args.relax_seconds)
    finally:
        summarize(cases, args.output_dir)


if __name__ == "__main__":
    main()
