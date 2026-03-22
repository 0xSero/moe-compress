#!/usr/bin/env python3
"""
Benchmark a model variant: lm-eval (100 samples each), SWE-bench style,
terminal/shell command generation, speed, and memory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--label", required=True, help="e.g. base_fp8, reap_20pct")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--patch-config", action="store_true",
                   help="Patch mistral4->mistral in config.json")
    return p.parse_args()


def log(msg, log_path):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def patch_config(model_path: Path):
    """Patch mistral4 -> mistral for transformers compat."""
    cfg_path = model_path / "config.json"
    bak = model_path / "config.json.bak"
    shutil.copy2(cfg_path, bak)
    with open(cfg_path) as f:
        cfg = json.load(f)
    if cfg.get("text_config", {}).get("model_type") == "mistral4":
        cfg["text_config"]["model_type"] = "mistral"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    return bak


def restore_config(model_path: Path):
    bak = model_path / "config.json.bak"
    if bak.exists():
        shutil.move(str(bak), str(model_path / "config.json"))


# ── lm-eval benchmarks ──────────────────────────────────────────────
def run_lm_eval(model_path, output_dir, label, limit, log_path):
    tasks = "arc_challenge,gsm8k,humaneval,hellaswag,winogrande,truthfulqa_mc2"
    results_dir = output_dir / "lm_eval"
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True,device_map=auto",
        "--tasks", tasks,
        "--limit", str(limit),
        "--batch_size", "1",
        "--output_path", str(results_dir),
        "--log_samples",
    ]
    log(f"lm-eval: {tasks} (limit={limit})", log_path)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0

    log(f"lm-eval done: {elapsed:.0f}s (exit={result.returncode})", log_path)
    if result.returncode != 0:
        log(f"lm-eval stderr: {result.stderr[-500:]}", log_path)

    # Parse results
    results_file = None
    for f in sorted(results_dir.rglob("results.json")):
        results_file = f
    if results_file:
        with open(results_file) as f:
            return json.load(f), elapsed
    return {"raw_stdout": result.stdout[-2000:]}, elapsed


# ── SWE-bench style eval ────────────────────────────────────────────
SWE_PROMPTS = [
    {
        "id": "swe_01",
        "prompt": "Given a Django project with the following error in the test suite:\n\n```\nAssertionError: 301 != 200\n```\n\nThe test expects a 200 OK response from `/api/users/` but gets a 301 redirect. The URL configuration is:\n```python\nurlpatterns = [\n    path('api/users', views.UserListView.as_view()),\n]\n```\n\nWhat is the bug and what is the fix? Provide the corrected code.",
        "expected_keywords": ["trailing slash", "api/users/", "APPEND_SLASH"],
    },
    {
        "id": "swe_02",
        "prompt": "A Python package's `setup.py` has:\n```python\ninstall_requires=['requests>=2.0', 'numpy']\n```\nBut `pip install -e .` fails with `ModuleNotFoundError: No module named 'numpy'` even though numpy is installed. The project uses a `pyproject.toml` with `build-system.requires = ['setuptools']`. What's wrong?",
        "expected_keywords": ["build", "requires", "numpy", "pyproject.toml", "build-system"],
    },
    {
        "id": "swe_03",
        "prompt": "This React component renders an infinite loop:\n```jsx\nfunction UserProfile({ userId }) {\n  const [user, setUser] = useState(null);\n  useEffect(() => {\n    fetch(`/api/users/${userId}`).then(r => r.json()).then(setUser);\n  });\n  return <div>{user?.name}</div>;\n}\n```\nExplain the bug and provide the fix.",
        "expected_keywords": ["dependency array", "useEffect", "[userId]", "missing"],
    },
    {
        "id": "swe_04",
        "prompt": "A Rust function panics at runtime:\n```rust\nfn parse_config(input: &str) -> Config {\n    let parts: Vec<&str> = input.split(':').collect();\n    Config {\n        host: parts[0].to_string(),\n        port: parts[1].parse().unwrap(),\n    }\n}\n```\nInput: `\"localhost\"`. Fix this to handle missing port gracefully, defaulting to 8080.",
        "expected_keywords": ["get(1)", "unwrap_or", "Option", "default", "8080"],
    },
    {
        "id": "swe_05",
        "prompt": "A SQL migration fails:\n```sql\nALTER TABLE users ADD COLUMN email VARCHAR(255) NOT NULL;\n```\nError: `ERROR: column \"email\" of relation \"users\" contains null values`\nThe table has 10M existing rows. Provide the correct migration.",
        "expected_keywords": ["DEFAULT", "NOT NULL", "backfill", "SET NOT NULL"],
    },
]

TERMINAL_PROMPTS = [
    {
        "id": "term_01",
        "prompt": "Write a bash one-liner that finds all Python files modified in the last 24 hours that contain 'TODO' comments, and outputs the file path and line number of each TODO.",
        "expected_keywords": ["find", "mtime", "grep", "-n", "TODO", "*.py"],
    },
    {
        "id": "term_02",
        "prompt": "Write a shell command to show the top 10 largest directories under /var/log, sorted by size, in human-readable format.",
        "expected_keywords": ["du", "sort", "head", "-h"],
    },
    {
        "id": "term_03",
        "prompt": "Write a bash command that monitors a log file in real-time and sends a desktop notification whenever a line containing 'ERROR' appears.",
        "expected_keywords": ["tail", "-f", "grep", "ERROR", "notify-send"],
    },
    {
        "id": "term_04",
        "prompt": "Write a command to find all Docker containers that exited with a non-zero exit code in the last hour and show their names and exit codes.",
        "expected_keywords": ["docker", "ps", "filter", "exited", "format"],
    },
    {
        "id": "term_05",
        "prompt": "Write a bash script that creates a tarball of all git-tracked files in the current repository, excluding anything in .gitignore, named backup-YYYYMMDD.tar.gz.",
        "expected_keywords": ["git", "ls-files", "tar", "date", ".tar.gz"],
    },
]


def run_custom_eval(model_path, output_dir, label, log_path):
    """Run SWE-bench style + terminal command benchmarks using the model directly."""
    from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

    log("Loading model for custom eval...", log_path)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    log(f"Model loaded: {time.time()-t0:.1f}s", log_path)

    results = {"swe_bench": [], "terminal": [], "speed": {}}

    def generate(prompt, max_new=512):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new,
                do_sample=False, temperature=1.0,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # SWE-bench eval
    log("Running SWE-bench style eval (5 prompts)...", log_path)
    for item in SWE_PROMPTS:
        t0 = time.time()
        response = generate(item["prompt"], max_new=1024)
        elapsed = time.time() - t0
        hits = sum(1 for kw in item["expected_keywords"] if kw.lower() in response.lower())
        score = hits / len(item["expected_keywords"])
        results["swe_bench"].append({
            "id": item["id"], "score": score, "hits": hits,
            "total_keywords": len(item["expected_keywords"]),
            "time_s": round(elapsed, 2),
            "response_preview": response[:300],
        })
        log(f"  {item['id']}: {score:.0%} ({hits}/{len(item['expected_keywords'])} keywords) {elapsed:.1f}s", log_path)

    # Terminal eval
    log("Running terminal command eval (5 prompts)...", log_path)
    for item in TERMINAL_PROMPTS:
        t0 = time.time()
        response = generate(item["prompt"], max_new=512)
        elapsed = time.time() - t0
        hits = sum(1 for kw in item["expected_keywords"] if kw.lower() in response.lower())
        score = hits / len(item["expected_keywords"])
        results["terminal"].append({
            "id": item["id"], "score": score, "hits": hits,
            "total_keywords": len(item["expected_keywords"]),
            "time_s": round(elapsed, 2),
            "response_preview": response[:300],
        })
        log(f"  {item['id']}: {score:.0%} ({hits}/{len(item['expected_keywords'])} keywords) {elapsed:.1f}s", log_path)

    # Speed benchmarks
    log("Running speed benchmarks...", log_path)

    # TTFT: 512 token prompt, measure time to first token
    prompt_512 = "Explain the concept of " + " ".join(["distributed systems"] * 50)
    inputs = tokenizer(prompt_512, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()
    ttft = time.time() - t0
    results["speed"]["ttft_512_s"] = round(ttft, 3)
    log(f"  TTFT (512 prompt): {ttft*1000:.0f}ms", log_path)

    # Generation throughput: generate 256 tokens
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    torch.cuda.synchronize()
    gen_time = time.time() - t0
    gen_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    results["speed"]["gen_256_time_s"] = round(gen_time, 3)
    results["speed"]["gen_256_tok_per_s"] = round(gen_tokens / gen_time, 2)
    log(f"  Gen 256 tokens: {gen_time:.1f}s ({gen_tokens/gen_time:.1f} tok/s)", log_path)

    # Prefill throughput: 4096 token prompt
    long_prompt = " ".join(["The quick brown fox jumps over the lazy dog."] * 500)
    inputs_long = tokenizer(long_prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = model(input_ids=inputs_long["input_ids"], attention_mask=inputs_long["attention_mask"])
    torch.cuda.synchronize()
    prefill_time = time.time() - t0
    prefill_tokens = inputs_long["input_ids"].shape[1]
    results["speed"]["prefill_4096_time_s"] = round(prefill_time, 3)
    results["speed"]["prefill_4096_tok_per_s"] = round(prefill_tokens / prefill_time, 1)
    log(f"  Prefill 4096: {prefill_time:.2f}s ({prefill_tokens/prefill_time:.0f} tok/s)", log_path)

    # Memory
    results["memory"] = {}
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.max_memory_allocated(i) / 1e9
        results["memory"][f"gpu_{i}_peak_gb"] = round(alloc, 2)
    total_vram = sum(v for v in results["memory"].values())
    results["memory"]["total_peak_gb"] = round(total_vram, 2)
    log(f"  Peak VRAM: {total_vram:.1f} GB", log_path)

    del model
    import gc; gc.collect(); torch.cuda.empty_cache()

    return results


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"bench_{args.label}.log"

    log(f"{'='*60}", log_path)
    log(f"BENCHMARK: {args.label}", log_path)
    log(f"Model: {model_path}", log_path)
    log(f"{'='*60}", log_path)

    patched = False
    if args.patch_config:
        patch_config(model_path)
        patched = True

    try:
        all_results = {"label": args.label, "model_path": str(model_path)}

        # 1. Custom eval (SWE + terminal + speed + memory)
        custom = run_custom_eval(model_path, output_dir, args.label, log_path)
        all_results["swe_bench"] = custom["swe_bench"]
        all_results["terminal"] = custom["terminal"]
        all_results["speed"] = custom["speed"]
        all_results["memory"] = custom["memory"]

        # Summary scores
        swe_avg = sum(r["score"] for r in custom["swe_bench"]) / len(custom["swe_bench"])
        term_avg = sum(r["score"] for r in custom["terminal"]) / len(custom["terminal"])
        log(f"\nSWE-bench avg: {swe_avg:.0%}", log_path)
        log(f"Terminal avg: {term_avg:.0%}", log_path)

        # 2. lm-eval
        lm_results, lm_time = run_lm_eval(model_path, output_dir, args.label, args.limit, log_path)
        all_results["lm_eval"] = lm_results
        all_results["lm_eval_time_s"] = lm_time

        # Save
        out_file = output_dir / f"results_{args.label}.json"
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)
        log(f"\nResults saved: {out_file}", log_path)

    finally:
        if patched:
            restore_config(model_path)

    log("BENCHMARK COMPLETE", log_path)


if __name__ == "__main__":
    main()
