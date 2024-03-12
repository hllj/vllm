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
--model Qwen-7B-Chat-AWQ-Marlin \
--quantization marlin \
--trust-remote-code \
--dtype bfloat16 \