#!/bin/bash

set -e

echo "Running Q7 Cheetah Hyperparameter Search"

Consider B sizes 10000, 30000, 50000
Consider learning rates 0.005, 0.01, 0.02
Use try different rtg  and nn_baseline settings
for B in 10000 30000 50000
do
    for LR in 0.005 0.01 0.02
    do
        for RTG in "" "-rtg"
        do
            for NN_BASELINE in "" "--nn_baseline"
            do
                echo "Running with B=$B LR=$LR RTG=$RTG NN_BASELINE=$NN_BASELINE"
                python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $B -lr $LR $RTG $NN_BASELINE --exp_name q4_b$B\_lr$LR$RTG$NN_BASELINE
            done
        done
    done
done
