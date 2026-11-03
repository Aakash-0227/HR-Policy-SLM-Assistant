
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Project configuration
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = ROOT_DIR / "experiments" / "lora_adapter" / "final"

SYSTEM_PROMPT = (
    "You are an HR Policy and Employee Support Assistant. "
    "Answer clearly and professionally. Do not invent company policies, "
    "employee records, leave balances, or payroll information. "
    "If specific information is unavailable, direct the employee to HR "
    "or the official company policy."
)

app = FastAPI(
    title="HR Policy & Employee Support Assistant",
    description="API powered by a Qwen SLM fine-tuned with LoRA.",
    version="1.0.0"
)

# Request validation
class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)

class ChatResponse(BaseModel):
    answer: str
    model: str

tokenizer = None
model = None

@app.on_event("startup")
def load_model():
    global tokenizer, model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    print("Fine-tuned HR SLM loaded successfully.")

@app.get("/")
def home():
    return {
        "message": "HR Policy Assistant API is running",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.question}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            key: value.to("cuda")
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        generated = output[0][inputs["input_ids"].shape[1]:]

        answer = tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

        return ChatResponse(
            answer=answer,
            model=MODEL_NAME
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(error)}"
        )
