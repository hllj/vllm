# python services/vllm/api_server.py \
# --host 0.0.0.0 \
# --port 8000 \
# --disable-log-requests \
# --model Qwen/Qwen-7B-Chat \
# --tokenizer Qwen/Qwen-7B-Chat \
# --trust-remote-code \
# --dtype bfloat16 \

python services/vllm/api_server.py \
    --host 0.0.0.0 \
    --port 8000 \
    --disable-log-requests \
    --model /root/AutoAWQ/mistral-instruct-v0.2-awq-marlin \
    --quantization awq \
    --enforce-eager \
    --trust-remote-code \
    --dtype float16 \
