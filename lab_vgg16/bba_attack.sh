#!/usr/bin/env bash

for model in `ls ../cp_res34`;
do
    # echo "model:" ${model} >> vgg16_ft_nloss_search.txt 
    python real_black_box.py --model-checkpoint ../cp_res34/${model} >> bba_attack_result_res34.txt 
done