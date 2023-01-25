#!/bin/bash

for config in ./configs/fcctc/exploratory/*.json
do
  CUDA_VISIBLE_DEVICES=1 python3 align_fcctc.py "$config"
done
