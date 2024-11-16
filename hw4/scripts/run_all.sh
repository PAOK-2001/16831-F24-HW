#!/bin/bash
set -e

set -x

echo "Running Q1 experiments"
sh /scripts/run_q1.sh
echo "Running Q2 experiments"
sh /scripts/run_q2.sh
echo "Running Q3 experiments"
sh /scripts/run_q3.sh
echo "Running Q4 experiments"
sh /scripts/run_q4.sh
echo "Running Q5 experiments"
sh /scripts/run_q5.sh

