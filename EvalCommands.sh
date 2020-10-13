#!/bin/bash

python LargeSizeEval.py \
    /data/datasets/yaoyuh/StereoData/bigresolution/AirSimCaptured \
    --file-list /data/datasets/yaoyuh/StereoData/bigresolution/AirSimCaptured/Files.csv \
    --loadmodel weights/final-768px.tar \
    --outdir LargeSize_Resize0.25 \
    --testres 0.25 \
    --max-disp 1024 \
    --max-num 0 \


