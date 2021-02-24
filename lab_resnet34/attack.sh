#!/usr/bin/env bash

for model in `ls ../cp_rebuttal_new`;
do
    echo "model:" ${model} >> resPGD_new.txt 
    python main_verify.py --model-checkpoint ../cp_rebuttal_new/${model} >> resPGD_new.txt 
done 