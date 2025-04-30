#!/bin/bash
source /import/ml-sc-scratch6/lang/trl-vlm-poc/venv/bin/activate

ACTUAL_CACHE=/import/ml-sc-scratch6/lang/cache/

export HF_DATASET_CACHE=${ACTUAL_CACHE}
export HF_MODULES_CACHE=${ACTUAL_CACHE}
export HF_HOME=${ACTUAL_CACHE}

FORCE_TORCHRUN=1 python train.py > /import/ml-sc-scratch6/lang/trl-vlm-poc/train_output.log 2>&1
