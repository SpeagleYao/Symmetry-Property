#!/usr/bin/env bash

for alpha in `echo 0.6`;
do
    for delta in `echo 0.6 0.8 1.0`
    do
        echo "alpha:" ${alpha}
        echo "delta:" ${delta}
        python train_ours.py --alpha ${alpha} --delta ${delta}
    done
done