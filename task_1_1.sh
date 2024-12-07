proxychains4 -q python task_1_1.py \
    -l 128 \
    -b 16 \
    -D cuda:1 \
    -t eval_throughput \
    -q original \
    -m facebook/opt-1.3b \
    --use_cache