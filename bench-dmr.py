from __future__ import annotations

import argparse
import json
import sys
import traceback
import requests
from statistics import mean, stdev
import random 

# Benchmarking Tool for Local Docker Model Runner
# This benchmark-tool is based very loosely on 
#   llama.cpp's server/bench.py and llama-bench

### ToDo - this ia an old version, port the newer bench-llama.py to this

###
### Helper-Functions for the benchmark
###
def model_get_token_cardinality(args):
    # Get the token cardinality from the model metadata
    # For now, we just return a dummy value
    return 256

def generate_prompt(args):
    # Generate a prompt for the benchmark consisting of tokens
    # **** ToDo: generate a prompt of args.n_prompt tokens ****
    prompt = "Say this is a test. On the other hand, it might not be a good test, with a lott of jumbled words ..." \
    
    # Randomly shuffle the next elements and add them in the middle
    prompt1 = "the quick brown fox jumps over the lazy dog "
    prompt1 += "a small red bird flies through the silent sky "
    prompt1 += "bright stars shine above the quiet town at night "
    prompt1 += "a happy child runs along the green field "
    prompt1 += "cold wind blows across the empty road "
    prompt1 += "a yellow cat sleeps near the warm fire "
    prompt1 += "early morning light hits the mountain peak "
    prompt1 += "gentle rain falls on the dusty path "
    prompt1 += "a loud train moves past the quiet station "
    prompt1 += "the clever fox watches from the tall grass "
    prompt1 += "a busy bee buzzes around the blooming flower "
    prompt1 += "white clouds drift across the blue sky "
    prompt1 += "a quiet river flows beside the old bridge "
    prompt1 += "the brave knight rides through the dark forest "
    prompt1 += "a curious dog sniffs around the wooden fence "
    prompt1 += "green leaves rustle in the autumn breeze "
    prompt1 += "a silver fish swims under the calm water "
    prompt1 += "soft snow covers the narrow trail "
    prompt1 += "a strong wind shakes the rusty gate "
    prompt1 += "the kind farmer feeds the hungry animals "
    words = prompt1.split()
    random.shuffle(words)
    prompt += ' '.join(words)
    
    prompt += " Just write a long story if it were a test. A real story, not just fill-words."
    #for i in range(args.n_prompt):
        # append tokens 

    return prompt


def server_benchmark_request(args,prompt):
    # Send a request to the server to benchmark the model
    url = f"http://{args.host}:{args.port}/engines/v1/completions"
    header = { "Content-Type": "application/json" }
    data = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.n_gen,
        "temperature": 1
    }
    # Convert the data to JSON format
    data = json.dumps(data)
    # Send the request
    response = requests.post(url, data=data, headers=header)
    if response.status_code != 200:
        raise RuntimeError(f"bench: server request failed with status code {response.status_code}: {response.text}")
    # Parse the JSON response
    response_data = response.json()
    if 'error' in response_data:
        raise RuntimeError(f"bench: server request failed with error: {response_data['error']}")
    return response_data
    

###
### Run the benchmark
###
def run_benchmark(args, results):

    print("Starting benchmark server-calls ", end="", flush=True)
    # we do args.repetitions iterations of the benchmark
    pp_timings = []
    tg_timings = []
    for iteration in range(args.repetitions):
        result=server_benchmark_request(args,generate_prompt(args))
        # measures: "tokens_predicted", "tokens_evaluated", "timings/prompt_per_token_ms", "timings/predicted_per_token_ms"
        if ('timings' in result.keys()) and ('prompt_per_token_ms' in result['timings'].keys()) and ('predicted_per_token_ms' in result['timings'].keys()):
            pp_timings.append(1000/(result['timings'].get('prompt_per_token_ms')))
            tg_timings.append(1000/(result['timings'].get('predicted_per_token_ms')))
        else:
            raise RuntimeError(f"bench: server did not return correct timings in {result}")
        print(".", end="", flush=True)
    
    # calculate the average of the timings, prompt/predicted_per_token_ms -> tokens_per_second
    results['pp_avg'] = mean(pp_timings)
    results['pp_avg_stdev'] = stdev(pp_timings)
    results['tg_avg'] = mean(tg_timings)
    results['tg_avg_stdev'] = stdev(tg_timings)
    print("", flush=True)

###
### print the results in .md format
###
def print_results(args, results):
    print("")
    print(f"| model                     | test |          t/s |")
    print(f"|---------------------------|------|--------------|")
    print(f"| {args.model} | pp{args.n_prompt} | {results['pp_avg']:.2f}  ± {results['pp_avg_stdev']:.2f} |")
    print(f"| {args.model} | tg{args.n_gen} | {results['tg_avg']:.2f} ± {results['tg_avg_stdev']:.2f} |")
    print("\n")

###
### main
###
def main(args_in: list[str] | None = None) -> None:

    print("\nBenchmarking Tool for Local Docker Model Runner V0.5\n")

    parser = argparse.ArgumentParser(description="Local AI-server benchmarking")

    # parameterd for running the local server
    parser.add_argument("--host", type=str, help="Server listen host", default="localhost")
    parser.add_argument("--port", type=int, help="Server listen port", default="12434")
    parser.add_argument('-m', "--model", type=str, help="Model name", default="ai/qwq:32B-Q4_K_M")

    # parameters for the benchmark
    parser.add_argument("-r", "--repetitions", type=int, help="Number of times to repeat each test", default=5)
    parser.add_argument("-p", "--n-prompt", type=int, help="Prompt-length to test", default=128) # should be 512
    parser.add_argument("-n", "--n-gen", type=int, help="Tokens to generate", default=128)

    args = parser.parse_args(args_in)

    # Benchmarking and print the results
    try:
        results = {}
        run_benchmark(args, results)
        print_results(args, results)
    except Exception:
        print("bench: error :")
        traceback.print_exc(file=sys.stdout)


if __name__ == '__main__':
    main()
