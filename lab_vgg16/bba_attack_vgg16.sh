#!/usr/bin/env bash

for model in `ls ../cp_vgg16`;
do
    # echo "model:" ${model} >> vgg16_ft_nloss_search.txt 
    python real_black_box.py --model-checkpoint ../cp_vgg16/${model} >> bba_attack_result_vgg16.txt 
done