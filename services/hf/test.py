from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)

# use bf16
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()

# System
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    }
]

# 1st turn
messages.append({
    "role": "user",
    "content": "Xin chào"
})

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print('1st chatml template', text)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print('1st turn response', response)

messages.append({
    "role": "assistant",
    "content": response
})

# 2nd turn 
messages.append({
    "role": "user",
    "content": "Đưa ra các số chia hết cho 3 và 5 bé hơn 100."
})

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print('2nd chatml template', text)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print('2nd turn response', response)