#!/bin/bash

for task in eval_throughput eval_gpu_memory
do
    for quantization in original int8 int4 int2
    do
        for len in 64 128 256 512
        do
            echo "$quantization - $len - $task"
            proxychains4 -q python task_1_1.py \
                -l $len \
                -b 16 \
                -D cuda:1 \
                -t $task \
                -q $quantization \
                -m facebook/opt-125m \
                -d data/data.txt
                # --use_cache
        done
    done
done