python benchmarks/benchmarks_throughput.py \
--backend vllm \
--dataset benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
--input-len 1024 \
--output-len 1024 \
--model Qwen/Qwen-7B-Chat \
--tokenizer Qwen/Qwen-7B-Chat \
--dtype bfloat16 \
--trust-remote-code \
--n 1 \
