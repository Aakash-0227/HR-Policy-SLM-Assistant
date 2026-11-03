
# Llama 3.2 1B - Baseline Experiment

## Model Information
- Model: meta-llama/Llama-3.2-1B-Instruct
- Parameters: Approximately 1.23 billion
- Execution: Local GPU inference
- GPU: NVIDIA RTX 3050 Laptop GPU
- Precision: FP16
- Framework: PyTorch
- CUDA: 12.8

## Test Case 1
Question: How can an employee apply for sick leave?

## Observed Results
The model suggested:
- Contacting the manager or supervisor
- Submitting a sick leave request
- Checking the employee portal
- Providing medical documentation if required

## Performance Metrics
- Inference Time: 10.91 seconds
- Peak GPU Memory: 2379.23 MB
- GPU Execution: Successful

## Evaluation
- Response generation: Successful
- Answer relevance: Partial
- Policy correctness: Not verified
- Hallucination risk: Present

## Limitations
No actual company HR policy documents were provided.
The generated answer is not verified company policy.

## Conclusion
Llama 3.2 successfully performed local GPU inference.
Its performance will be compared with the other candidate models.
