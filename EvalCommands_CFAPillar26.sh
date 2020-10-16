#!/bin/bash

python LargeSizeEval.py \
    /data/datasets/yaoyuh/StereoData/Hardware4K/CFAPillar \
    --file-list /data/datasets/yaoyuh/StereoData/Hardware4K/CFAPillar/File26.csv \
    --pc-q /data/datasets/yaoyuh/StereoData/Hardware4K/CFAPillar/Q.dat \
    --loadmodel weights/final-768px.tar \
    --outdir Hardwar4K_CFAPillar_Resize0.25 \
    --testres 0.25 \
    --max-disp 1024 \
    --max-num 0 \


