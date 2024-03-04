# python api_server.py \
# --host 0.0.0.0 \
# --port 8000 \
# --disable-log-requests \
# --model Qwen/Qwen-7B-Chat \
# --tokenizer Qwen/Qwen-7B-Chat \
# --trust-remote-code \
# --dtype bfloat16 \

# Scheduler

python api_server.py \
--host 0.0.0.0 \
--port 8000 \
--disable-log-requests \
--max-num-batched-tokens 16384 \
--model Qwen/Qwen-7B-Chat \
--tokenizer Qwen/Qwen-7B-Chat \
--trust-remote-code \
--dtype bfloat16 \