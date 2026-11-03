
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading Qwen tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

messages = [
    {
        "role": "system",
        "content": "You are an HR Policy and Employee Support Assistant."
    },
    {
        "role": "user",
        "content": "How can I update my tax declaration?"
    },
    {
        "role": "assistant",
        "content": "Use the organization's designated tax declaration or payroll portal."
    }
]

formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)

tokens = tokenizer(
    formatted_text,
    return_tensors="pt"
)

print("\nFormatted conversation:")
print(formatted_text)

print("\nToken count:", tokens["input_ids"].shape[1])
print("Tokenizer test completed successfully!")
