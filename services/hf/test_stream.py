from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import TextIteratorStreamer

from threading import Thread

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)

# use bf16
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()

# stream
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

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
    "content": "Xin chào, hãy đếm từ 1 đến 100"
})

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print('1st chatml template', text)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
thread = Thread(target=model.generate, kwargs=generation_kwargs)

thread.start()
generated_text = ""
for new_text in streamer:
    generated_text += new_text
    print(generated_text)