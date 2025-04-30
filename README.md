# Replicating LlamaFactory VLM Training with TRL

## Goal
Demonstrate that multimodal supervised fine-tuning (vision-language) is possible using TRL as a baseline alternative to LlamaFactory.

Note: this is a PoC, not full feature parity.

## Core Components
- `train.py`: entry point for training, including trainer integration with TRL `SFTTrainer`.
- `run.sh`: launch script for training
- `data/`: building blocks for data preparation
- `model/`: building blocks for model preparation
- `params.py`: parameters grouped by how they are used
- `requirements.txt`: dependencies
- `llamafactory_refs`: components ported from LlamaFactory (modifications are commented with prefix `[LlamaFactory]`)

## Results
Output directory: `/import/ml-sc-scratch6/lang/llama_3.2_checkpoints_gpu_trl`

## Epoch=1, Reduced dataset, Frozen Layers, 1 GPU
```
    "total_flos": 30938066691072.0,
    "train_loss": 1.723783254623413,
    "train_runtime": 81.9167,
    "train_samples_per_second": 0.098,
    "train_steps_per_second": 0.049
```

## Gap Analysis
See [comparison](https://docs.google.com/spreadsheets/d/1Qbzbla4IF7Z7qlXPtWY7YaMPOzYRzgmsXyMSnpfyk7g/edit?usp=sharing)
