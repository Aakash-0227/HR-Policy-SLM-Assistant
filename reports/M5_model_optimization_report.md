# M5 – Model Optimization Report
## HR Policy & Employee Support Assistant

### 1. Objective
Evaluate a generation-setting optimization on the LoRA fine-tuned Qwen2.5-0.5B-Instruct model using the same 15 held-out HR FAQ test questions.

### 2. Optimization applied
The maximum number of newly generated tokens was reduced from 120 to 80. Greedy decoding was retained (`do_sample=False`) and the KV cache was enabled (`use_cache=True`). Two warm-up runs were performed for each configuration, and the measurement order was alternated across test questions.

### 3. Benchmark results

| Metric | Standard configuration | Optimized configuration |
|---|---:|---:|
| Maximum new tokens | 120 | 80 |
| Average latency | 2.051 sec | 2.006 sec |
| Average peak GPU memory | 957.36 MB | 957.36 MB |
| Average generated tokens | 30.67 | 30.67 |
| Test questions | 15 | 15 |

The measured latency difference is **2.19% lower** for the 80-token configuration (a difference of 0.045 seconds per answer on average).

### 4. Interpretation
- Both configurations generated the same average number of tokens (30.67) on this test set.
- Average peak GPU memory was unchanged at 957.36 MB.
- The 80-token configuration showed a small measured latency reduction of 2.19%.
- Because the generated answers averaged fewer than 80 tokens, the lower maximum-token limit did not constrain the average output length in this run.
- The observed latency difference is small and should not be presented as a large or guaranteed speed improvement. More repeated runs would be needed to establish whether it is consistently faster.

### 5. Quality and safety considerations
The optimized configuration must be checked for answer completeness and truncation before it is selected for deployment. A lower token limit can cut off longer answers. The benchmark JSON stores the generated answers for both settings so they can be compared against the reference answers. This report does not claim a quality improvement from the token-limit change.

### 6. Conclusion
The generation optimization experiment was completed on all 15 held-out questions. Reducing `max_new_tokens` from 120 to 80 produced a small observed latency reduction from 2.051 to 2.006 seconds, while average peak GPU memory and average generated token count remained unchanged. The 80-token setting is a candidate configuration, subject to review of answer completeness. The measured improvement is modest and should be interpreted cautiously.

### 7. Reproducibility and evidence
- Script: `src/optimization/benchmark_generation_optimization.py`
- Results: `experiments/generation_optimization_results.json`
- Model: `Qwen/Qwen2.5-0.5B-Instruct` with the trained LoRA adapter in `experiments/lora_adapter/final`
- Test data: `data/splits/test.json`
