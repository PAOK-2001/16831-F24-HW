#!/bin/bash

for B_SIZE in 100 500 1000
do
    for LR in 0.01 0.02 0.05 0.1
    do
        python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 99 -l 2 -s 64 -b $B_SIZE -lr $LR -rtg --exp_name q2_b"$B_SIZE"_r"$LR"_rtg
    done
done
