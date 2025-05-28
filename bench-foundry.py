from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import requests
from statistics import mean, stdev
from foundry_local import FoundryLocalManager

# Benchmarking Tool for Microsoft Foundry Local
# This benchmark-tool is based very loosely on 
#   llama.cpp's server/bench.py and llama-bench

# server-request helper-Function for the benchmark
# it uses the generic open-ai API 'v1/chat/completions' endpoint
def server_benchmark_request(endpoint, model, prompt, n_gen):
    # Build the request URL and headers
    url = f"{endpoint}/chat/completions"
    header = { "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [ { 
            "role": "user", 
            "content": prompt 
        } ], 
        "max_tokens": n_gen,
        "max_completion_token": n_gen,
        "temperature": 1
    }
    # Convert data to JSON format
    data = json.dumps(data)
    # Send + time the request
    start_time = time.time()
    response = requests.post(url, data=data, headers=header)
    time_taken = time.time() - start_time
    if response.status_code != 200:
        raise RuntimeError(f"bench: server request failed with status code {response.status_code}: {response.text}")
    # Parse the JSON response
    response_data = response.json()
    return time_taken


# Run the benchmark
def run_benchmark(endpoint, model, n_gen, repetitions, results):

    print("Starting benchmark server-calls .", end="", flush=True)
    
    # we can't measure the time for pp, so we use a fixed prompt and measure the time for a single token generation overhead
    prompt = "Write a long story."
    # we do a warmup-run to ignore the overhead of a first request
    time_warmup=server_benchmark_request(endpoint, model, prompt, 100)
    print(".", end="", flush=True)
    # then we measure the overhead for a few token generations
    n_overhead=60  # number of tokens to generate for the overhead measurement
    time_overhead=server_benchmark_request(endpoint, model, prompt, n_overhead)
    
    # we do args.repetitions iterations of the benchmark
    tg_timings = []
    for iteration in range(repetitions):
        print(".", end="", flush=True)
        time_run=server_benchmark_request(endpoint, model, prompt, n_gen + n_overhead)  # +1 to account for measured oveverhead token
        # measures time taken for request
        tg_timings.append(1/((time_run - time_overhead) / n_gen))  # convert to tokens per second

    # calculate the average of the timings, prompt/predicted_per_token_ms -> tokens_per_second
    results['tg_avg'] = mean(tg_timings)
    results['tg_avg_stdev'] = stdev(tg_timings)
    print(".", flush=True)


### print the results in .md format
def print_results(model, n_gen, results):
    print("")
    print(f"| model                        |  test |          t/s |")
    print(f"|------------------------------|-------|--------------|")
    print(f"| {model} | tg{n_gen} | {results['tg_avg']:.2f} ± {results['tg_avg_stdev']:.2f} |")
    print("\n")


###
### main
###
def main(args_in: list[str] | None = None) -> None:

    print("\nMBenchmarking Tool for Microsoft Foundry Local V1.0\n")

    parser = argparse.ArgumentParser(description="Local AI-server benchmarking")

    # parameterd for running the local server
    parser.add_argument('-m', "--model", type=str, help="Model name", default="Phi-4-mini-reasoning-qnn-npu")

    # parameters for the benchmark
    parser.add_argument("-r", "--repetitions", type=int, help="Number of times to repeat each test", default=5)
    parser.add_argument("-n", "--n-gen", type=int, help="Tokens to generate", default=128)

    args = parser.parse_args(args_in)
    
    # Create an instance of the FoundryLocalManager anddownload the model
    print("bench: initializing Local Foundry and loading the model.", flush=True)
    try:
        manager = FoundryLocalManager(alias_or_model_id=args.model, bootstrap=True)
        model_info = manager.download_model(args.model)
        model_info = manager.load_model(args.model)
    except Exception:
        print("bench: error initializing Local Foundry:")
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
    print("    ... done.", flush=True)

    # Benchmarking and print the results
    try:
        results = {}
        run_benchmark(manager.endpoint, manager.get_model_info(args.model).id, args.n_gen, args.repetitions, results)
        print_results(manager.get_model_info(args.model).id, args.n_gen, results)
    except Exception:
        print("bench: error :")
        traceback.print_exc(file=sys.stdout)

    # cleanup
    manager.unload_model(args.model)


if __name__ == '__main__':
    main()
