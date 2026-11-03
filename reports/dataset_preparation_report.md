
# M3 – Dataset Preparation Report

## 1. Project
HR Policy & Employee Support Assistant

## 2. Dataset
File: data/raw/hr_faq_dataset.jsonl

The dataset contains sample HR-related question-and-answer records.

## 3. Dataset Categories
- Leave management
- Attendance
- Payroll
- Employee onboarding
- Company policies

## 4. Dataset Validation
Validation script: src/data/validate_dataset.py

Result:
- Validation errors: 0
- Validation status: PASSED

## 5. Dataset Splitting
Script: src/data/split_dataset.py

Split ratio:
- Training: 70%
- Validation: 15%
- Testing: 15%

Random seed: 42

## 6. Split Results

| Dataset | Records |
|---|---:|
| Training | 7 |
| Validation | 1 |
| Testing | 2 |
| Total | 10 |

## 7. Output Files
- data/splits/train.json
- data/splits/validation.json
- data/splits/test.json

## 8. Evidence
- screenshots/M3_Dataset_Preparation/02_dataset_validation.png
- screenshots/M3_Dataset_Preparation/03_dataset_split_completed.png

## 9. Limitations
The current dataset contains only 10 records.
It is sufficient for initial pipeline testing but not for reliable model fine-tuning or final evaluation.

A larger and more diverse dataset is required before training.

## 10. Status
Initial dataset preparation, validation and splitting completed.

## 11. Baseline Model Performance Comparison

Three SLMs were evaluated using the prepared HR test dataset.

| Model | Average Inference Time | Peak GPU Memory |
|---|---:|---:|
| Qwen2.5-0.5B-Instruct | 9.90 seconds | 956.96 MB |
| TinyLlama-1.1B-Chat | 3.12 seconds | 2118.12 MB |
| Llama-3.2-1B-Instruct | 3.08 seconds | 2380.71 MB |

### Observations

- Qwen used the least GPU memory but had the highest inference time.
- TinyLlama provided a balanced speed and memory result.
- Llama 3.2 achieved the lowest inference time but consumed more GPU memory.

### Evaluation Limitation

The current held-out test set contains only two records.
Therefore, these results are preliminary and require evaluation
with a larger and more diverse test dataset.

### Evidence

- screenshots/M3_Dataset_Preparation/06_baseline_performance_comparison.png

### Status

Baseline inference and initial performance comparison completed.
