# HR Policy & Employee Support Assistant

An AI-powered HR question-answering assistant built using a Small Language Model (SLM), LoRA fine-tuning and FastAPI.

## 1. Project Overview

The **HR Policy & Employee Support Assistant** is an AI application designed to answer common employee questions related to:

- Leave policies
- Attendance procedures
- Payroll FAQs
- Employee onboarding
- Company policies

The project uses **Qwen2.5-0.5B-Instruct**, a Small Language Model (SLM), adapted using Low-Rank Adaptation (LoRA) fine-tuning.

The adapted model is deployed through a FastAPI application, allowing users to submit HR-related questions and receive generated responses through an API.

The project uses a synthetic HR question-and-answer dataset for development and evaluation. It does not represent an official company's HR policy.

## 2. Project Objectives

- Research and compare different Small Language Models.
- Select a suitable SLM based on model size, license, hardware requirements and capabilities.
- Prepare and validate a domain-specific HR question-and-answer dataset.
- Split the dataset into training, validation and testing sets.
- Evaluate the selected model before fine-tuning.
- Adapt the model using LoRA fine-tuning.
- Compare baseline and fine-tuned model performance.
- Optimize text generation settings and measure performance.
- Deploy the fine-tuned model using FastAPI.
- Implement input validation and API error handling.

## 3. Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python 3.13 |
| Base Model | Qwen2.5-0.5B-Instruct |
| Model Framework | Hugging Face Transformers |
| Fine-tuning | PEFT, LoRA |
| Deep Learning | PyTorch |
| Dataset Processing | JSON, Python |
| API Framework | FastAPI |
| API Server | Uvicorn |
| Request Validation | Pydantic |
| Model Evaluation | Python benchmarking scripts |
| Version Control | Git and GitHub |

## 4. Small Language Model Research

Three models were researched and compared during model selection:

| Model | Parameters | License | Context Length |
|---|---:|---|---:|
| Qwen2.5-0.5B-Instruct | 0.5B | Apache 2.0 | 32K |
| TinyLlama-1.1B-Chat | 1.1B | Apache 2.0 | 2K |
| Llama 3.2 1B Instruct | 1B | Llama Community License | 128K |

Qwen2.5-0.5B-Instruct was selected because its smaller model size makes it suitable for experimentation on limited GPU memory.

## 5. Dataset Preparation

A synthetic HR question-and-answer dataset was prepared for model adaptation.

The expanded dataset contains 100 records across five categories:

| Category | Records |
|---|---:|
| Leave | 25 |
| Attendance | 19 |
| Payroll | 19 |
| Onboarding | 14 |
| Company Policies | 23 |
| **Total** | **100** |

### Dataset splitting

The dataset was divided into three subsets:

| Dataset | Records |
|---|---:|
| Training | 69 |
| Validation | 16 |
| Testing | 15 |
| **Total** | **100** |

The dataset was validated to check record structure, unique IDs and duplicate questions.

The training records were formatted using the Qwen chat template.

## 6. Model Fine-tuning Using LoRA

The selected base model was adapted using Low-Rank Adaptation (LoRA).

LoRA updates a small number of trainable parameters instead of updating the entire base model. This reduces the training requirements compared with full fine-tuning.

### Training configuration

| Parameter | Value |
|---|---|
| Base Model | Qwen2.5-0.5B-Instruct |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.05 |
| Target Modules | q_proj, v_proj |
| Epochs | 3 |
| Learning Rate | 0.0002 |
| Batch Size | 1 |
| Gradient Accumulation | 4 |
| Precision | FP16 |
| Maximum Training Length | 256 tokens |

### Training results

| Epoch | Training Loss |
|---|---:|
| Epoch 1 | 2.8471 |
| Epoch 2 | 2.3937 |
| Epoch 3 | 2.1241 |

Trainable parameters: 540,672 (approximately 0.1093% of the base model parameters).

Peak GPU memory during training: approximately 1193.44 MB.

The training loss decreased across the three epochs, indicating improvement on the training data. This alone does not establish generalization to unseen company policies.

## 7. Model Evaluation

The baseline model and LoRA fine-tuned model were evaluated using the same 15 held-out test questions.

### Manual answer-quality evaluation

Five criteria were used:

1. Accuracy
2. Relevance
3. Completeness
4. Avoidance of unsupported claims
5. Professionalism

Each criterion was scored from 0 to 2, giving a maximum score of 10 per answer.

| Model | Average Quality Score |
|---|---:|
| Baseline Model | 6.00 / 10 |
| Fine-tuned Model | 8.27 / 10 |

The fine-tuned model improved the average manual quality score by 2.27 points.

These are manually reviewed scores, not automated benchmark metrics. The evaluation uses a small synthetic test dataset.

### Performance comparison

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| Average Latency | 11.983 sec | 2.02 sec |
| Average Peak GPU Memory | 957.09 MB | 957.36 MB |
| Test Questions | 15 | 