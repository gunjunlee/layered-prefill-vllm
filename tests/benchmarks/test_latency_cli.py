# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import csv
import json
import subprocess
from types import SimpleNamespace

import pytest

from benchmarks import benchmark_layered_prefill as layered

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


@pytest.mark.benchmark
def test_bench_latency():
    command = [
        "vllm",
        "bench",
        "latency",
        "--model",
        MODEL_NAME,
        "--input-len",
        "32",
        "--output-len",
        "1",
        "--enforce-eager",
        "--load-format",
        "dummy",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


@pytest.fixture
def layered_config():
    return json.loads(layered.DEFAULT_CONFIG.read_text())


def test_layered_figure3_preserves_all_panels_and_both_models(layered_config, tmp_path):
    args = layered.parse_args(["--output-dir", str(tmp_path / "out")])
    cases, skipped = layered.make_cases(layered.read_config(layered_config), args)
    assert not skipped
    assert len(cases) == 48
    expected = {
        "3a": (
            "Qwen/Qwen3-30B-A3B",
            "arxiv",
            [1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            10000,
            125,
        ),
        "3b": (
            "unsloth/gpt-oss-20b-BF16",
            "arxiv",
            [2.1, 2.3, 2.5, 2.7, 2.9, 3.1],
            10000,
            100,
        ),
        "3c": (
            "Qwen/Qwen3-30B-A3B",
            "sharegpt",
            [4.0, 4.2, 4.4, 4.6, 4.8, 5.0],
            5000,
            125,
        ),
        "3d": (
            "unsloth/gpt-oss-20b-BF16",
            "sharegpt",
            [5.8, 6.0, 6.2, 6.4, 6.6, 6.8],
            5000,
            100,
        ),
    }
    for panel, (model, dataset, rates, ttft, tbt) in expected.items():
        selected = [c for c in cases if c["sources"][0]["panel"] == panel]
        assert len(selected) == 12
        for budget in (512, 8192):
            paired = [
                c for c in selected if c["engine"]["max_num_batched_tokens"] == budget
            ]
            assert [c["workload"]["request_rate"] for c in paired] == rates
            for case in paired:
                assert case["engine"]["model"] == model
                assert case["engine"]["dtype"] == "bfloat16"
                assert case["engine"]["tensor_parallel_size"] == 2
                assert case["engine"]["pipeline_parallel_size"] == 1
                assert case["workload"]["dataset"] == dataset
                assert case["workload"]["ttft_slo_ms"] == ttft
                assert case["workload"]["tbt_slo_ms"] == tbt
                assert case["workload"]["num_prompts"] == 600
                command = layered.benchmark_command(case, args)
                assert command[command.index("--burstiness") + 1] == "1.0"
                assert "--goodput" not in command  # Mean TPOT is not the paper's TBT.


def test_layered_config_keeps_full_token_budget_and_local_groups(
    layered_config, tmp_path
):
    args = layered.parse_args(
        ["--output-dir", str(tmp_path / "out"), "--models", "gpt-oss", "--pp", "2"]
    )
    cases, _ = layered.make_cases(layered.read_config(layered_config), args)
    engine = next(c["engine"] for c in cases if c["engine"]["num_layer_groups"] > 1)
    assert engine["num_layer_groups"] == 12
    assert engine["pipeline_parallel_size"] == 2
    assert engine["max_num_batched_tokens"] == 8192
    assert engine["enable_chunked_prefill"]
    assert not engine["enforce_eager"]
    assert engine["compilation_config"]["cudagraph_capture_sizes"][-1] == 8192
    args.enforce_eager = True
    eager_cases, _ = layered.make_cases(layered.read_config(layered_config), args)
    assert all(c["engine"]["enforce_eager"] for c in eager_cases)
    assert all("compilation_config" not in c["engine"] for c in eager_cases)


def test_layered_latency_deduplicates_rates_but_preserves_models(
    layered_config, tmp_path
):
    args = layered.parse_args(
        ["--output-dir", str(tmp_path / "out"), "--benchmark", "latency"]
    )
    cases, _ = layered.make_cases(layered.read_config(layered_config), args)
    assert len(cases) == 4
    assert all(len(c["sources"]) == 12 for c in cases)
    assert all(c["workload"]["input_len"] == 8192 for c in cases)
    assert "--request-rate" not in layered.benchmark_command(cases[0], args)


def test_layered_config_is_independent_of_reference_results(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "benchmark_results.csv").write_text("not a csv")
    layered.main(
        [
            "--source-dir",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert len(manifest["cases"]) == 48
    assert not manifest["skipped"]
    assert manifest["experiment_config"]["paper"]["figure"] == 3
    assert manifest["environment"]["VLLM_USE_V2_MODEL_RUNNER"] == "1"


def test_layered_resume_rejects_changed_workload(tmp_path):
    options = [
        "--output-dir",
        str(tmp_path / "out"),
        "--benchmark",
        "latency",
        "--dry-run",
    ]
    layered.main(options)
    with pytest.raises(ValueError, match="manifest differs"):
        layered.main([*options, "--input-len", "4096", "--resume"])


def test_layered_override_requires_one_model(tmp_path):
    with pytest.raises(SystemExit):
        layered.parse_args(["--output-dir", str(tmp_path), "--model", "local-bf16"])
    args = layered.parse_args(
        ["--output-dir", str(tmp_path), "--models", "gpt-oss", "--model", "local-bf16"]
    )
    cases, _ = layered.make_cases(layered.read_config(args.experiment_config), args)
    assert len(cases) == 24
    assert all(c["engine"]["model"] == "local-bf16" for c in cases)


def test_layered_slo_rejects_one_slow_token_even_when_mean_tpot_passes():
    result: dict = {
        "ttfts": [0.1, 0.1, 11.0, 0.1],
        "itls": [[0.01, 0.2], [0.1], [0.01], []],
        "errors": ["", "", "", ""],
        "output_lens": [3, 2, 2, 1],
    }
    metrics = layered.paper_slo_metrics(
        result, {"num_prompts": 4, "ttft_slo_ms": 10000, "tbt_slo_ms": 125}
    )
    assert metrics == {
        "slo_attained": 2,
        "slo_attainment_pct": 50,
        "ttft_attainment_pct": 75,
        "tbt_attainment_pct": 75,
    }
    result["itls"][0] = []
    with pytest.raises(ValueError, match="Missing inter-token"):
        layered.paper_slo_metrics(
            result, {"num_prompts": 4, "ttft_slo_ms": 10000, "tbt_slo_ms": 125}
        )


def test_layered_failed_requests_are_not_reported_as_success(
    layered_config, tmp_path, monkeypatch
):
    args = layered.parse_args(
        ["--output-dir", str(tmp_path / "out"), "--warmup-seconds", "0"]
    )
    cases, _ = layered.make_cases(layered.read_config(layered_config), args)
    case = cases[0]

    def incomplete_benchmark(command, log, env, timeout):
        layered.write_json(log.parent / "result.json", {"completed": 599})

    monkeypatch.setattr(layered, "run_command", incomplete_benchmark)
    with pytest.raises(RuntimeError, match="Only 599 requests completed"):
        layered.run_case(case, args, {})
    assert not layered.is_complete(args.output_dir / case["id"])
    layered.summarize(cases[:2], args.output_dir)
    with (args.output_dir / "summary.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert [row["status"] for row in rows] == ["failed", "pending"]


@pytest.fixture
def layered_arxiv_samples():
    class Dataset:
        data = [
            SimpleNamespace(prompt="too long", expected_output_len=1),
            SimpleNamespace(prompt="exact fit", expected_output_len=2),
            SimpleNamespace(prompt="output too long", expected_output_len=3),
            SimpleNamespace(prompt="next article", expected_output_len=1),
        ]

        def sample(self, tokenizer, num_requests):
            return self.data[:num_requests]

    def tokenizer(prompt):
        lengths = {
            "too long": 67567,
            "exact fit": 32766,
            "output too long": 32766,
            "next article": 1234,
        }
        return SimpleNamespace(input_ids=range(lengths[prompt]))

    return Dataset(), tokenizer


def test_layered_arxiv_replaces_over_context_requests_without_truncation(
    layered_arxiv_samples,
):
    dataset, tokenizer = layered_arxiv_samples
    original = list(dataset.data)
    requests, skipped = layered.sample_dataset(dataset, tokenizer, "arxiv", 2, 32768)
    assert requests == [
        {"prompt": "exact fit", "output_tokens": 2},
        {"prompt": "next article", "output_tokens": 1},
    ]
    assert skipped == 2
    assert dataset.data == original


def test_layered_arxiv_exhaustion_reports_count_instead_of_duplicating_requests(
    layered_arxiv_samples,
):
    dataset, tokenizer = layered_arxiv_samples
    with pytest.raises(
        ValueError, match="Expected 3 valid arxiv requests, got 2; skipped 2"
    ):
        layered.sample_dataset(dataset, tokenizer, "arxiv", 3, 32768)
