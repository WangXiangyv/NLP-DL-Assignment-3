#!/bin/bash

for task in eval_gpu_memory eval_throughput
do
    for cache in no_kv_cache golden_kv_cache simple_kv_cache
    do
        for len in 64 128 256 512
        do
            echo "$len - $cache - $task"
            proxychains4 -q python task_1_2.py \
                -l $len \
                -b 16 \
                -D cuda:1 \
                -t $task \
                -c $cache \
                -d data/data.txt
        done
    done
done