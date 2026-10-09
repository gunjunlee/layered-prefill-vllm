# Benchmarks

This directory used to contain vLLM's benchmark scripts and utilities for performance testing and evaluation.

## Contents

- **Serving benchmarks**: Scripts for testing online inference performance (latency, throughput)
- **Throughput benchmarks**: Scripts for testing offline batch inference performance
- **Specialized benchmarks**: Tools for testing specific features like structured output, prefix caching, long document QA, request prioritization, and multi-modal inference
- **Dataset utilities**: Framework for loading and sampling from various benchmark datasets (ShareGPT, HuggingFace datasets, synthetic data, etc.)

## Usage

For detailed usage instructions, examples, and dataset information, see the [Benchmark CLI documentation](https://docs.vllm.ai/en/latest/benchmarking/cli/#benchmark-cli).

For full CLI reference see:

- <https://docs.vllm.ai/en/latest/cli/bench/latency.html>
- <https://docs.vllm.ai/en/latest/cli/bench/serve.html>
- <https://docs.vllm.ai/en/latest/cli/bench/throughput.html>

## Layered prefill: Figure 3

`benchmark_layered_prefill.py` uses the explicit
[`configs/layered_prefill/figure3.json`](configs/layered_prefill/figure3.json)
configuration, transcribed from
[Figure 3 of arXiv:2510.08055v2](https://arxiv.org/html/2510.08055v2#S5.F3).
It no longer reads experiment settings from result CSVs or active Python code.

| Panel | Model | Dataset | Request rates (req/s) | TTFT / TBT SLO (ms) |
| --- | --- | --- | --- | --- |
| 3a | Qwen3-30B-A3B | arXiv | 1.3, 1.4, 1.5, 1.6, 1.7, 1.8 | 10000 / 125 |
| 3b | GPT-OSS-20B | arXiv | 2.1, 2.3, 2.5, 2.7, 2.9, 3.1 | 10000 / 100 |
| 3c | Qwen3-30B-A3B | ShareGPT | 4.0, 4.2, 4.4, 4.6, 4.8, 5.0 | 5000 / 125 |
| 3d | GPT-OSS-20B | ShareGPT | 5.8, 6.0, 6.2, 6.4, 6.6, 6.8 | 5000 / 100 |

Each panel runs both chunked and layered prefill: **48 serving experiments**.
The paper uses **H100 80GB x2 with NVLink, TP=2, BF16**, with Poisson arrivals.
Running on A100 is a hardware override, even with the same workload config.

```bash
# Preview all 48 experiments without GPUs or dataset downloads.
.venv/bin/python benchmarks/benchmark_layered_prefill.py \
  --benchmark serve --output-dir layered-figure3 --dry-run

# Execute. Waits until both selected GPUs use <= 500 MiB.
.venv/bin/python benchmarks/benchmark_layered_prefill.py \
  --benchmark serve --output-dir layered-figure3 --gpus 0,1

# Resume; completed cases are skipped.
.venv/bin/python benchmarks/benchmark_layered_prefill.py \
  --benchmark serve --output-dir layered-figure3 --gpus 0,1 --resume

# Select only GPT-OSS (24 cases); add --dataset arxiv for just panel 3b.
.venv/bin/python benchmarks/benchmark_layered_prefill.py \
  --models gpt-oss --benchmark serve --output-dir layered-figure3-gpt

# Synthetic batch latency: four engine configs, not Figure 3 SLO curves.
.venv/bin/python benchmarks/benchmark_layered_prefill.py \
  --benchmark latency --input-len 8192 --output-len 1 --batch-size 1 \
  --output-dir layered-figure3-latency
```

`--config PATH` selects an edited config. `--models qwen` / `--models gpt-oss`
filters models without replacing them. `--model PATH` requires one selected
model and overrides its checkpoint, for example:

```bash
.venv/bin/python benchmarks/benchmark_layered_prefill.py \
  --models gpt-oss --model /path/to/gpt-oss-20b-bf16 \
  --output-dir layered-figure3-local-gpt --dry-run
```

GPT-OSS defaults to `unsloth/gpt-oss-20b-BF16`, whose
[configuration](https://huggingface.co/unsloth/gpt-oss-20b-BF16/blob/main/config.json)
is unquantized BF16. This matches the weight precision in Section 5.1.
The official `openai/gpt-oss-20b` checkpoint contains MXFP4 expert weights;
`--dtype bfloat16` alone does not convert those weights to BF16. The checkpoint
choice is explicit in the config and manifest.

Values not specified in Figure 3 are labeled in the config's `paper.sources`:
max sequences 256, context 32768, GPU memory fraction 0.85, and measurement
length `min(600 requests, 600 seconds * request rate)` come from the reference
benchmark script. Its warmup and cooldown are 60 seconds each. Change these
with `--warmup-seconds` and `--relax-seconds`.

Serving uses the reference repository **only for dataset sampling**, including
its ShareGPT trace lengths and tokenizer templates. `--source-dir` defaults
to `../layered-prefill`. It exports shared custom JSONL files for paired
schedulers and does not reshuffle or template them a second time. Dataset
sampling needs `datasets`, `pandas`, `Pillow`, `transformers`, and access to the
source datasets. Latency and dry runs do not need that repository.

For arXiv, requests whose tokenized prompt plus requested output exceed the
context limit are skipped, and subsequent articles fill the requested count
(600 by default). Prompts and output lengths are not shortened. Each model's
chunked and layered runs share the same filtered JSONL. The number of rejected
requests is printed and saved in `datasets/*.metadata.json`; selection fails
if the source does not contain enough valid requests.

Each output directory contains the config and its hash in `manifest.json`,
commands, logs, detailed JSON results, and `summary.csv`. Serving results include
`slo_attainment_pct`, `ttft_attainment_pct`, and `tbt_attainment_pct`, using
Table 5 thresholds. A request passes only if its TTFT and **every measured
inter-token latency** pass. This is different from vLLM's mean-TPOT goodput.
Failed/incomplete runs remain failed. Use a **new output directory** when
switching from the old recorded/active script or changing the runner or any config.

Current implementation differences are recorded in the manifest:

- Both schedulers use Model Runner V2, synchronous scheduling, no prefix caching,
  breakable prefill CUDA graphs, and full decode CUDA graphs. `torch.compile`
  is disabled. Attention remains eager in breakable prefill graphs.
  `--enforce-eager` disables graphs for both.
  Layered prefill (`--num-layer-groups > 1`) automatically selects V2 and rejects
  `VLLM_USE_V2_MODEL_RUNNER=0`. The benchmark explicitly selects V2 for both modes.
- Chunked uses a 512-token budget; layered uses 8192. Fixed layer groups
  (Qwen 16, GPT-OSS 12) follow the reference code. The paper adapts group count
  to input length; the current runner retains one batch through its groups.
  It does not reproduce the paper's continuous decode schedule. These configs
  reproduce the Figure 3 workload matrix, not a claim of identical results.
- Groups partition each PP rank. Small requests share the full
  `max_num_batched_tokens` budget; PP never divides that budget. `--tp`, `--pp`,
  and `--ep` are explicit parallelism overrides. For two-GPU PP use `--tp 1 --pp 2`.
- Energy and average decode-batch-size instrumentation are not included.
  `--benchmark latency` deduplicates rates/datasets into four model/scheduler
  combinations and measures synthetic batch completion time.

CUDA graph capture sizes cover powers of two through each token budget.
More groups and capture sizes increase startup time and graph memory.
