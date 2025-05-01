#!/bin/bash
source /import/ml-sc-scratch6/lang/trl-vlm-poc/venv/bin/activate

ACTUAL_CACHE=/import/ml-sc-scratch6/lang/cache/

export HF_DATASET_CACHE=${ACTUAL_CACHE}
export HF_MODULES_CACHE=${ACTUAL_CACHE}
export HF_HOME=${ACTUAL_CACHE}
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=2

torchrun --nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=$NPROC_PER_NODE train.py > /import/ml-sc-scratch6/lang/llama_3.2_checkpoints_gpu_trl/train_output.log 2>&1
# python -m pdb train.py
