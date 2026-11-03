# HR Policy & Employee Support Assistant
## M2 – SLM Model Research, Benchmarking and Selection

### 1. Project Model Candidates

Three Small Language Models were selected for baseline testing.

| Model | Parameters | Developer |
|---|---:|---|
| Qwen2.5-0.5B-Instruct | 0.5B | Alibaba |
| TinyLlama-1.1B-Chat | 1.1B | TinyLlama |
| Llama-3.2-1B-Instruct | 1B | Meta |

### 2. Model Benchmark Environment

- Operating System: Windows 11
- Programming Language: Python 3.13.13
- Deep Learning Framework: PyTorch
- Model Library: Hugging Face Transformers
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- GPU VRAM: 4 GB
- System RAM: 8 GB

### 3. Benchmark Test Question

How can an employee apply for sick leave?

All three models received the same question to compare their response quality and inference performance.

### 4. Benchmark Results

| Model | Inference Time | Peak GPU Memory |
|---|---:|---:|
| Qwen2.5-0.5B-Instruct | 26.12 seconds | 957.45 MB |
| TinyLlama-1.1B-Chat | 10.24 seconds | 2118.15 MB |
| Llama-3.2-1B-Instruct | 8.41 seconds | 2381.28 MB |

### 5. Answer Quality Observations

**Qwen2.5-0.5B-Instruct**
- Generated a structured answer.
- Explained reviewing the leave policy, completing the application form and submitting the application.
- The generated response ended before completing the final step.

**TinyLlama-1.1B-Chat**
- Generated a simple and concise answer.
- Mentioned checking the leave policy, contacting HR and providing employee information.
- Included documentation suggestions that may not apply to every company.

**Llama-3.2-1B-Instruct**
- Generated a clear and professional answer.
- Explained checking the leave policy and submitting a request.
- The generated response was incomplete.

### 6. Final Model Selection

**Selected Initial Base Model: Qwen/Qwen2.5-0.5B-Instruct**

#### Reason for Selection

Qwen2.5-0.5B-Instruct was selected as the initial base model because it consumed the lowest GPU memory during the baseline benchmark.

Its smaller model size and lower memory consumption make it a practical starting point for experimentation on the available NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM.

#### Benchmark Performance

- Inference Time: 26.12 seconds
- Peak GPU Memory: 957.45 MB
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU

#### Identified Limitations

- Higher inference time compared with the other tested models.
- Generated answer was incomplete.
- HR-specific answer quality requires further improvement.
- Model responses must be grounded in verified HR policy documents.

### 7. Planned Improvements

The following improvements will be explored in later project milestones:

1. HR domain dataset preparation.
2. LoRA-based model adaptation.
3. Retrieval-Augmented Generation (RAG).
4. Model quantization and inference optimization.
5. Evaluation using held-out HR test questions.
6. Comparison of baseline and final model performance.

### 8. Model Selection Status

Status: Initial model selected based on baseline benchmark results.

The selection is provisional and will be evaluated again after domain adaptation, optimization and final testing.

### 9. Evidence Screenshots

The following benchmark evidence is maintained in the project:

- screenshots/M2_Model_Benchmark/01_benchmark_completed.png
- screenshots/M2_Model_Benchmark/02_model_metrics_comparison.png
- screenshots/M2_Model_Benchmark/03_model_answer_quality.png