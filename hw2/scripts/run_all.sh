#!/bin/bash
set -e

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 {q5|q6|q7|all}"
    exit 1
fi

run_q5() {
    echo "Running Q5"
    ./scripts/q5_run_small_batch.sh
    ./scripts/q5_run_large_batch.sh
}


run_q7() {
    echo "Running Q7"
    ./scripts/q7_exp3_lunar_landar.sh
    ./scripts/q7_cheetah.sh
}

run_q8() {
    echo "Running Q8"
    ./scripts/q8_gae.sh
    # scripts/q8.sh
}

for arg in "$@"; do
    case "$arg" in
        q5)
            run_q5
            ;;
        q6)
            run_q6
            ;;
        q7)
            run_q7
            ;;
        q8)
            run_q8
            ;;
        *)
            echo "Invalid option: $arg"
            echo "Usage: $0 {q5|q6|q7|q8|all}"
            exit 1
            ;;
    esac
done
