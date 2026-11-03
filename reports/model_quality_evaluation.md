# Model Quality Evaluation Report
## HR Policy & Employee Support Assistant

### 1. Evaluation objective

This evaluation compares the original Qwen2.5-0.5B-Instruct model with the LoRA fine-tuned version for the HR Policy & Employee Support Assistant. Both models were assessed on the same 15 held-out test questions using the corresponding reference answers.

### 2. Evaluation method

Each answer was manually reviewed against its reference answer using five criteria. Each criterion receives a score from 0 to 2, giving a maximum of 10 points per answer.

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Accuracy | Incorrect or misleading | Partly correct | Correct |
| Relevance | Does not answer the question | Partly relevant | Directly relevant |
| Completeness | Major information missing | Some important information missing | Covers key points in the reference |
| Avoidance of unsupported claims | Significant unsupported claims | Minor unsupported claims | No unsupported claims |
| Professional communication | Unclear or unprofessional | Understandable but needs improvement | Clear and professional |

The same five criteria were applied to the original and fine-tuned answers. Scores are manual judgments, not automated metrics. The detailed per-case scores and reviewer observations are in `manual_quality_evaluation_completed.xlsx` and `manual_quality_evaluation_completed.csv`.

### 3. Overall results

| Model | Average score | Test questions |
|---|---:|---:|
| Original Qwen2.5-0.5B-Instruct | 6.00/10 | 15 |
| LoRA fine-tuned Qwen2.5-0.5B-Instruct | 8.27/10 | 15 |

The fine-tuned model's mean manual score was higher by **2.27 points out of 10** on this test set.

### 4. Per-test-case results

| No. | Test ID | Category | Original | Fine-tuned |
|---:|---|---|---:|---:|
| 1 | HR032 | attendance | 8/10 | 9/10 |
| 2 | HR009 | leave_management | 8/10 | 9/10 |
| 3 | HR098 | company_policies | 10/10 | 10/10 |
| 4 | HR004 | leave_management | 6/10 | 7/10 |
| 5 | HR040 | attendance | 6/10 | 9/10 |
| 6 | HR059 | payroll | 2/10 | 6/10 |
| 7 | HR060 | payroll | 6/10 | 8/10 |
| 8 | HR036 | attendance | 6/10 | 9/10 |
| 9 | HR078 | company_policies | 6/10 | 7/10 |
| 10 | HR014 | leave_management | 1/10 | 7/10 |
| 11 | HR057 | payroll | 6/10 | 9/10 |
| 12 | HR079 | company_policies | 9/10 | 7/10 |
| 13 | HR073 | onboarding | 3/10 | 9/10 |
| 14 | HR018 | leave_management | 5/10 | 9/10 |
| 15 | HR065 | onboarding | 8/10 | 9/10 |

### 5. Observations

- The original model sometimes produced unsupported or irrelevant details, including invented process steps or document requirements.
- The fine-tuned model more often gave concise answers that directed employees to HR, managers, IT, or an approved organizational process.
- The fine-tuned model still omitted some details present in the references and occasionally added details that were not explicitly supported.
- Performance varied by question. A higher aggregate score does not mean every fine-tuned answer was fully correct.

### 6. Limitations

- The test set contains 15 synthetic/generic HR FAQ examples and does not represent an official organization's handbook or policy.
- The assessment is manual and involves reviewer judgment. Scores should be reviewed by the project owner before final submission.
- These results measure answer quality against the supplied reference answers; they do not establish factual correctness for any real employer's policies.
- The current report does not claim statistical significance or generalization beyond this small test set.

### 7. Conclusion

On the 15 held-out examples, the LoRA fine-tuned model achieved a higher average manual quality score than the original model (**8.27/10 versus 6.00/10**). The result suggests improvement on this synthetic HR FAQ task. Further validation with a larger, reviewed dataset and real approved HR policy documents would be needed before using the assistant for organization-specific guidance.

### 8. Related evidence

- `manual_quality_evaluation_completed.xlsx` — detailed scores, observations, and scoring rubric.
- `manual_quality_evaluation_completed.csv` — tabular evaluation data.
- `baseline_vs_finetuned_results.json` — generated answers and inference measurements.
- `answer_quality_comparison.md` — question-by-question answer comparison.
