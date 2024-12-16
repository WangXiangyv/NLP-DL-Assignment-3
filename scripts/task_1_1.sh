#!/bin/bash

for task in eval_gpu_memory eval_throughput
do
    for quantization in int8 int4 int2 original
    do
        for len in 64 128 256 512
        do
            echo "$quantization - $len - $task"
            proxychains4 -q python task_1_1.py \
                -l $len \
                -b 16 \
                -D cuda:0 \
                -t $task \
                -q $quantization \
                -m facebook/opt-125m \
                -d data/data.txt \
                --use_cache
        done
    done
done