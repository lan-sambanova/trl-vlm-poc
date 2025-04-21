from dataclasses import dataclass
from typing import Optional


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
