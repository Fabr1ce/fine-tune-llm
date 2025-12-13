import time
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


def benchmark_model(model_path, prompt, num_runs=5):
    """Benchmark model inference speed"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=50)
    
    times = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    tokens_per_sec = 100 / avg_time
    
    print(f"Average time: {avg_time:.2f}s")
    print(f"Tokens/sec: {tokens_per_sec:.2f}")
    
    return avg_time, tokens_per_sec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain machine learning")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    
    benchmark_model(args.model, args.prompt, args.runs)
