#!/usr/bin/env bash

for alpha in `echo 2.0 2.5 5.0 7.5 10.0 15.0`;
do
    for delta in `echo 0.08 0.1 0.12`
    do
        echo "alpha:" ${alpha}
        echo "delta:" ${delta}
        python train_ours.py --alpha ${alpha} --delta ${delta}
    done
done