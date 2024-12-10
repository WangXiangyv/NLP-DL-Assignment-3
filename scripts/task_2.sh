#!/bin/bash

proxychains4 -q python task_2.py \
    -o results/R-1-5.jsonl \
    -p Reflexion \
    -a sk-f5317b6ccb9341faa3beab36d4838cb1 \
    -u https://api.deepseek.com \
    -m deepseek-chat \
    -N 500 \
    -n 1