python benchmarks/benchmark_latency.py \
--model Qwen/Qwen-7B-Chat \
--tokenizer Qwen/Qwen-7B-Chat \
--input-len 1024 \
--output-len 1024 \
--batch-size 8 \
--n 1 \
--trust-remote-code \
--dtype bfloat16 \
--profile \
--profile-result-dir result/latency/baseline \
