#!/bin/bash

set -e

BSIZE=30000
LR=0.02

python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $BSIZE -lr $LR --exp_name q7_optimal

python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $BSIZE -lr $LR -rtg --exp_name q7_optimal_rtg

python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $BSIZE -lr $LR --nn_baseline --exp_name q7_optimal_nnbaseline

python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $BSIZE -lr $LR -rtg --nn_baseline --exp_name q7_optimal_rtg_nnbaseline