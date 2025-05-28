# bench-llm

This project is a Python port of a small functional subset of llama-bench.cpp from llama.cpp. It uses an openAI API v1 endpoint as the source for benchmarking.

Due to some backends not exposing tokens or the tokenizer, this benchmark only provides a simple tg benchmark.

The current status is that of work-in-progress for the folloring variants:

- bench-dmr.py    Benchmarking Tool for Local Docker Model Runner - old version, uses reported timings
- bench-foundry   Benchmarking Tool for Microsoft Foundry Local - includes startup/shutdown for Foundry
- bench-llama     Local openAI API (e.g. llama.cpp's llama-server) LLM Benchmarking Tool 

## How to run

1. Ensure you have Python 3.8+ and the llama.cpp llama-server installed.
2. Install the required Python packages from the `requirements.txt` file in an .venv or conda environment. Installing foundry_local is only required when using the Foundry variant.
3. Run the main benchmark script:

```text
python bench-<variant>.py [options]
```

### Server-options

```text
  -h, --help                show this help message and exit
  --endpoint                Server listen host endpoint
  -m, --model MODEL         Model filename (type .gguf)
```

### Benchmark-options

```text
  -r, --repetitions <n>     Repeat each test this many times to average the random prompts
  -n, --n-gen <n>           Tokens to generate for the benchmark
```
