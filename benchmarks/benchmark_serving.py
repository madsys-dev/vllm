"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_hf_server.sh <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> --dataset <target_dataset> \
        --request-rate <request_rate>
"""
import argparse
import asyncio
from dataclasses import dataclass
import json
import random
import time
from typing import AsyncGenerator, List, Tuple
import tqdm

import aiohttp
import numpy as np
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []

async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    backend: str,
    api_url: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
) -> None:
    request_start_time = time.perf_counter()

    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        pload = {
            "prompt": '<dummy>',
            "prompt_token_ids": [1] * prompt_len,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": False,
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)

            # Re-send the request if it failed.
            if "error" not in output:
                break

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))


async def benchmark(
    backend: str,
    url: str,
    num_requests: int,
    sampler,
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
) -> 'Metrics':
    api_url = url + '/generate'
    metrics_url = url + '/metrics'
    
    # Warmup
    for _idx in tqdm.tqdm(range(num_requests)):
        prompt_len, total_len = sampler()
        task = asyncio.create_task(send_request(backend, api_url, prompt_len, total_len - prompt_len,
                                                best_of, use_beam_search))
        await asyncio.sleep(1.0 / request_rate)

    start_metrics = await Metrics.fetch(metrics_url)
    for _idx in tqdm.tqdm(range(num_requests)):
        prompt_len, total_len = sampler()
        task = asyncio.create_task(send_request(backend, api_url, prompt_len, total_len - prompt_len,
                                                best_of, use_beam_search))
        await asyncio.sleep(1.0 / request_rate) 
    metrics = await Metrics.fetch(metrics_url)
    return metrics - start_metrics

class RequestGenerator:
    def __init__(self, trace_path: str):
        import csv
        reader = csv.reader(open(trace_path))
        next(reader)
        traces = []
        for row in reader:
            prompt_len, total_len = map(int, row)
            if total_len > 1024:
                continue
            traces.append((prompt_len, total_len))
        self.traces = traces

    def __call__(self) -> Tuple[int, int] | None:
        return random.choice(self.traces)


@dataclass
class Metrics:
    iters: int 
    generated_tokens: int 
    time: float

    @staticmethod
    async def fetch(url: str) -> 'Metrics':
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
                output = b"".join(chunks).decode()
        import re
        iters = re.search(r'^vllm_bench:iters ([0-9.]+)', output, flags=re.MULTILINE).group(1)
        generated_tokens = re.search(r'^vllm_bench:generated_tokens ([0-9.]+)', output, flags=re.MULTILINE).group(1)
        return Metrics(float(iters), float(generated_tokens), time.perf_counter())

    def __sub__(self, other: 'Metrics') -> 'Metrics':
        return Metrics(self.iters - other.iters, self.generated_tokens - other.generated_tokens, 
                          self.time - other.time)

def main(args: argparse.Namespace):
    # print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    url = f"http://{args.host}:{args.port}"
    request_generator = RequestGenerator(args.dataset)

    metrics = asyncio.run(benchmark(args.backend, url, args.num_prompts, request_generator, args.best_of,
                          args.use_beam_search, args.request_rate))
    if args.csv:
        print(args.request_rate,args.num_prompts, metrics.time, metrics.iters, metrics.generated_tokens,sep=',')
    else:
        print('Elapsed time: {} s'.format(metrics.time))
        print('Iterations: {}'.format(metrics.iters))
        print('Generated tokens: {}'.format(metrics.generated_tokens))
        print('Throughput: {} tok/s'.format(metrics.generated_tokens / metrics.time))
        print('Latency: {} s'.format(metrics.time / metrics.iters))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm", "tgi"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--best-of", type=int, default=1,
                        help="Generates `best_of` sequences per prompt and "
                             "returns the best one.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                             "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process to synthesize "
                             "the request arrival times.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code', action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument('--csv', action='store_true', help='print results in csv format')
    args = parser.parse_args()
    main(args)
