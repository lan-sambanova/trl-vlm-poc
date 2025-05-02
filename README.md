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

## Epoch=1, 8 Examples, Frozen Layers, 1 GPU
```
    "total_flos": 30938066691072.0,
    "train_loss": 1.723783254623413,
    "train_runtime": 81.9167,
    "train_samples_per_second": 0.098,
    "train_steps_per_second": 0.049
```

## Epoch=1, 128 Examples, Frozen Layers, 2 GPUs
```
   "total_flos": 2039011967238144.0,
    "train_loss": 1.2031240463256836,
    "train_runtime": 78.2295,
    "train_samples_per_second": 1.636,
    "train_steps_per_second": 0.026
```

128 examples distributed across 2 GPUs:
```
World size: 2
Per device train batch size: 2
Effective global train batch size: 64
Num of train steps: 2
```

Note: I was not able to run the full dataset due to some issues with some images of the dataset.
See: `/import/ml-sc-scratch6/lang/llama_3.2_checkpoints_gpu/train_output_full_dataset.log`

## Gap Analysis
See [comparison](https://docs.google.com/spreadsheets/d/1Qbzbla4IF7Z7qlXPtWY7YaMPOzYRzgmsXyMSnpfyk7g/edit?usp=sharing)
