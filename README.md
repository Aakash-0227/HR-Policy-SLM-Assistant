# HR Policy & Employee Support Assistant

## 1. Project Overview

The HR Policy & Employee Support Assistant is an AI-powered application built using a Small Language Model (SLM).

The application helps employees get answers to common HR-related questions about leave policies, attendance, payroll procedures, employee onboarding and company policies.

The system uses an SLM adapted for HR-related tasks and Retrieval-Augmented Generation (RAG) to retrieve relevant information from HR policy documents.

## 2. Project Objectives

- Build an AI assistant using a locally runnable Small Language Model.
- Prepare a domain-specific HR question-and-answer dataset.
- Adapt the selected model using LoRA fine-tuning.
- Optimize model performance using quantization.
- Integrate RAG for HR policy document retrieval.
- Evaluate baseline and adapted model performance.
- Deploy the application using FastAPI and Streamlit.

## 3. Application Modules

1. Leave Management Assistant
2. Attendance Policy Assistant
3. Payroll Support Assistant
4. Employee Onboarding Assistant
5. Company Policy Assistant

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Model Loading | Hugging Face Transformers |
| Model Adaptation | PEFT, LoRA |
| Dataset Processing | Pandas, JSONL |
| Optimization | Quantization |
| Backend | FastAPI |
| Frontend | Streamlit |
| Evaluation | Scikit-learn, Pytest |
| Version Control | Git and GitHub |

## 5. Project Architecture

Employee Input → FastAPI → Input Validation → Prompt Preparation → SLM Inference → Output Validation → Response

RAG will be implemented as a separate retrieval component to provide relevant HR policy context.

## 6. Project Structure

```text
HR-Policy-SLM-Assistant/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── data/
│   ├── model/
│   ├── evaluation/
│   ├── optimization/
│   └── api/
├── tests/
├── experiments/
├── reports/
├── requirements.txt
├── README.md
└── .gitignore
```

## 7. Implementation Milestones

1. Define the project use case.
2. Research and select an SLM.
3. Prepare the HR-specific dataset.
4. Build and evaluate the baseline model.
5. Adapt the model using LoRA.
6. Optimize model performance.
7. Evaluate the final model.
8. Deploy the application and prepare the final report.

## 8. Project Status

Currently in the initial project setup and planning phase.

Implementation, model selection, dataset preparation, training, evaluation and deployment will be completed progressively.

## 9. Expected Outcome

A working HR Policy & Employee Support Assistant that can answer employee questions using an adapted Small Language Model and relevant HR policy information.