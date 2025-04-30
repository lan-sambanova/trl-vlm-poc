from dataclasses import dataclass
from typing import List, Optional

import torch


# Ref: src/llamafactory/hparams/data_args.py
@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    template: Optional[str] = None
    dataset: Optional[str] = None
    eval_dataset: Optional[str] = None
    cutoff_len: int = 1024
    overwrite_cache: bool = False
    preprocessing_num_workers: Optional[int] = None
    preprocessing_batch_size: int = 1000
    max_samples: Optional[int] = None


# Ref: src/llamafactory/hparams/model_args.py
@dataclass
class ModelArguments:
    model_name_or_path: str
    use_fast_tokenizer: bool = True
    split_special_tokens: bool = False
    cache_dir: Optional[str] = None
    model_revision: str = "main"
    hf_hub_token: Optional[str] = None
    image_resolution: int = 512
    video_resolution: int = 128
    video_fps: float = 2.0
    video_maxlen: int = 64
    gradient_checkpointing: bool = True
    block_diag_attn: bool = False  # whether use block diag attn or not
    compute_dtype: Optional[torch.dtype] = None  # Torch dtype for computing model outputs, derived from `fp/bf16`. Do not specify it.

# Ref: src/llamafactory/hparams/finetuning_args.py
@dataclass
class FinetuningArguments:
    stage: str
    finetuning_type: str
    freeze_vision_tower: bool = True  # Whether or not to freeze vision tower in MLLM training
    train_mm_proj_only: bool = False  # Whether or not to train the multimodal projectos for MLLM training
    use_llama_pro: bool = False  # Whether to make only the params in the expanded blocks trainable
    freeze_trainable_layers: int = 2  # Positive num means the last n layers are trainable
    freeze_trainable_modules: str = "all"  # Name(s) of trainable modules for freeze fine-tuning
    freeze_extra_modules: Optional[str] = None  # Name(s) of modules apart from hidden layers to be set as trainable
    pure_bf16: bool = False
    use_badam: bool = False  # Whether or not to use the BAdam (block-diag update params) optimizer

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.freeze_trainable_modules: List[str] = split_arg(self.freeze_trainable_modules)
        self.freeze_extra_modules: Optional[List[str]] = split_arg(self.freeze_extra_modules)
        self.freeze_vision_tower = self.freeze_vision_tower or self.train_mm_proj_only

        assert self.finetuning_type in ["lora", "freeze", "full"], "Invalid fine-tuning method."

        if self.train_mm_proj_only and self.finetuning_type != "full":
            raise ValueError("`train_mm_proj_only` is only valid for full training.")
