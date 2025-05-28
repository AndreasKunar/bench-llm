from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import requests
from statistics import mean, stdev

# Local openAI API (e.g. llama.cpp's llama-server) LLM Benchmarking Tool 
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
        "temperature": 1.9
    }
    # Convert data to JSON format
    data = json.dumps(data)
    # Send + time the request
    start_time = time.time()
    response = requests.post(url, data=data, headers=header)
    time_taken = time.time() - start_time
    if response.status_code != 200:
        raise RuntimeError(f"bench: server request failed with status code {response.status_code}: {response.text}")
    # Parse the JSON response to see if we got the expected number of tokens
    tokens_generated = response.json().get("usage").get("total_tokens", 0)
    return time_taken, tokens_generated  # return time taken and number of tokens generated


# Run the benchmark
def run_benchmark(endpoint, model, n_gen_target, repetitions, results):

    print("Preparing benchmark server-calls .", end="", flush=True)
    
    # we can't measure the time for pp, so we use a fixed prompt and measure the time for a single token generation overhead
    prompt = "Write a very long story."
    # we do a warmup-run to ignore the overhead of a first request
    time_warmup, n_warmup = server_benchmark_request(endpoint, model, prompt, 100)
    print(".", end="", flush=True)
    # then we measure the overhead for a few token generations
    n_overhead_target = 60  # target number of tokens to generate for the overhead measurement
    # generate the "overhead", but use the actually generated number of tokens
    time_overhead, n_overhead = server_benchmark_request(endpoint, model, prompt, n_overhead_target)
    print(".\nDoing benchmark server-calls .", end="", flush=True)
    
    # we do "repetitions" iterations of the benchmark
    tg_timings = []
    for iteration in range(repetitions):
        time_gen, n_gen = server_benchmark_request(endpoint, model, prompt, n_gen_target + n_overhead)
        # measures time taken for request, but only if more token than overhead were generated
        token_per_s= (n_gen - n_overhead) / (time_gen - time_overhead)   # time for the n_gen tokens minus the overhead
        if (n_gen > n_overhead):
            tg_timings.append(token_per_s)
            print(".", end="", flush=True)
        
    # calculate the average of the timings, prompt/predicted_per_token_ms -> tokens_per_second
    results['tg_avg'] = mean(tg_timings)
    results['tg_avg_stdev'] = stdev(tg_timings)
    print("", flush=True)


### print the results in .md format
def print_results(model, n_gen, results):
    print("")
    print(f"| model                          |  test |          t/s |")
    print(f"|--------------------------------|-------|--------------|")
    print(f"| {model} | tg{n_gen} | {results['tg_avg']:.2f} ± {results['tg_avg_stdev']:.2f} |")
    print("\n")


###
### main
###
def main(args_in: list[str] | None = None) -> None:

    print("\nLocal openAI API v1 LLM Endpoint Benchmarking Tool V1.0\n")

    parser = argparse.ArgumentParser(description="Local AI-server benchmarking")

    # parameters for running the local server
    parser.add_argument("--endpoint", type=str, help="Server listen host endpoint", default="http://127.0.0.1:8007/v1")
    parser.add_argument('-m', "--model", type=str, help="Model name", default="../models.llama.cpp/Microsoft/Phi-4-mini-reasoning-Q4_0.gguf")

    # parameters for the benchmark
    parser.add_argument("-r", "--repetitions", type=int, help="Number of times to repeat each test", default=5)
    parser.add_argument("-n", "--n-gen", type=int, help="Tokens to generate", default=128)

    args = parser.parse_args(args_in)
    model_name=os.path.basename(args.model)
    
    # Benchmarking and print the results
    try:
        results = {}
        run_benchmark(args.endpoint, model_name, args.n_gen, args.repetitions, results)
        print_results(model_name, args.n_gen, results)
    except Exception:
        print("bench: error :")
        traceback.print_exc(file=sys.stdout)


if __name__ == '__main__':
    main()
