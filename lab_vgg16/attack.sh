#!/usr/bin/env bash

for model in `ls ../checkpoint/vgg16_ft_nloss`;
do
    echo "model:" ${model} >> vgg16_ft_nloss_search.txt 
    python main_verify.py --model-checkpoint ../checkpoint/vgg16_ft_nloss/${model} >> vgg16_ft_nloss_search.txt 
done  