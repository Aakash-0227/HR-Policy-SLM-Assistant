
# TinyLlama 1.1B - Baseline Experiment

## Model Information
- Model: TinyLlama-1.1B-Chat-v1.0
- Parameters: 1.1 billion
- Execution: Local GPU inference
- GPU: NVIDIA RTX 3050 Laptop GPU
- Precision: FP16
- Framework: PyTorch
- CUDA: 12.8

## Test Case 1
Question: How can an employee apply for sick leave?

## Observed Results
The model generated an answer describing:
- Completing an application form
- Providing employee name and job title
- Mentioning the reason for leave
- Submitting supporting documents

## Performance Metrics
- Inference Time: 7.49 seconds
- Peak GPU Memory: 2115.91 MB
- GPU Execution: Successful

## Evaluation
- Response generation: Successful
- Answer relevance: Partial
- Policy correctness: Not verified
- Hallucination risk: Present

## Limitations
The model was tested without actual HR policy documents.
Its answer cannot be treated as verified company policy.

## Conclusion
TinyLlama successfully performed local inference on the RTX 3050.
Its response and resource usage will be compared with other candidate models.
