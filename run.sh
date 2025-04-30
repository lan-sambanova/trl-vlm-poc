#!/bin/bash

ACTUAL_CACHE=/import/ml-sc-scratch6/lang/cache/

export HF_DATASET_CACHE=${ACTUAL_CACHE}
export HF_MODULES_CACHE=${ACTUAL_CACHE}
export HF_HOME=${ACTUAL_CACHE}

python -m pdb train.py
