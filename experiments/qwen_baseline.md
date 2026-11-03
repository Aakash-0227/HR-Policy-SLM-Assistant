
# Qwen2.5-0.5B-Instruct - Baseline Experiment

## Model Information
- Model: Qwen2.5-0.5B-Instruct
- Parameters: 0.49 billion
- Execution: Local GPU inference
- GPU: NVIDIA RTX 3050 Laptop GPU
- Precision: FP16
- Framework: PyTorch
- CUDA: 12.8

## Test Case 1

Question:
How can an employee apply for sick leave?

## Observed Result

The model generated an answer describing eligibility criteria, an application form and an application process.

It also introduced unsupported conditions, including:
- At least 10 years of employment.
- No more than 3 days of absence in the previous year.

## Evaluation

- GPU inference: Successful
- Response generation: Successful
- Answer relevance: Partial
- Policy correctness: Not verified
- Hallucination: Observed

## Limitation

The model was tested without an HR policy document or retrieval system. Therefore, it generated generic information and unsupported eligibility conditions.

## Conclusion

The model successfully runs locally on the RTX 3050. However, it requires HR-specific grounding and further evaluation before being used for employee support.
