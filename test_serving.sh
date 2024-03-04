for i in 0 1 2
do
    python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model Qwen/Qwen-7B-Chat \
    --tokenizer Qwen/Qwen-7B-Chat \
    --trust-remote-code \
    --dataset benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 10 \
    --save-result \
    --exp-dir result/qwen-7b-chat \
    --exp-name scheduler-max-num-batched-tokens-16384 \
    --version $i
done