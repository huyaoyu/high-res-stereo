#!/bin/bash

# On Azure.
# DATA_ROOT=/data/datasets/yaoyuh/StereoData/Hardware4K/CFAPillar
# FILE_LIST=/data/datasets/yaoyuh/StereoData/Hardware4K/CFAPillar/File26.csv
# PC_Q=/data/datasets/yaoyuh/StereoData/Hardware4K/CFAPillar/Q.dat

# Local.
DATA_ROOT=/media/yaoyu/DiskE/Projects/NewStereo/Hardware4K/CFAPillar
FILE_LIST=/media/yaoyu/DiskE/Projects/NewStereo/Hardware4K/CFAPillar/File26.csv
PC_Q=/media/yaoyu/DiskE/Projects/NewStereo/Hardware4K/CFAPillar/Calibration/Q.dat

# python LargeSizeEval.py \
#     ${DATA_ROOT} \
#     --file-list ${FILE_LIST} \
#     --pc-q  ${PC_Q}\
#     --loadmodel weights/final-768px.tar \
#     --outdir Hardwar4K_CFAPillar_Resize0.25 \
#     --testres 0.25 \
#     --max-disp 1024 \
#     --max-num 0 \

python LargeSizeEval.py \
    ${DATA_ROOT} \
    --file-list ${FILE_LIST} \
    --pc-q  ${PC_Q}\
    --loadmodel weights/final-768px.tar \
    --outdir Hardwar4K_CFAPillar_Resize0.5 \
    --testres 0.5 \
    --max-disp 1024 \
    --max-num 0 \

