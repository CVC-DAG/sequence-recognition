#!/bin/bash

for config in ./configs/ctc/exploratory/*.json
do
  CUDA_VISIBLE_DEVICES=1 python3 align_ctc.py "$config"
done
