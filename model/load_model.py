from dataclasses import dataclass
from typing import Optional

from transformers import AutoConfig, AutoTokenizer, AutoProcessor

from model.model_utils import get_image_seqlen

@dataclass
class ModelArgs:
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

def _set_extra_attr(processor, tokenizer, args: ModelArgs):
    config = load_config(args)
    setattr(processor, "tokenizer", tokenizer)
    setattr(processor, "image_seqlen", get_image_seqlen(config))
    setattr(processor, "image_resolution", args.image_resolution)
    setattr(processor, "video_resolution", args.video_resolution)
    setattr(processor, "video_fps", args.video_fps)
    setattr(processor, "video_maxlen", args.video_maxlen)

def _get_init_kwargs(args: ModelArgs):
    return {
        "trust_remote_code": True,
        "cache_dir": args.cache_dir,
        "revision": args.model_revision,
        "token": args.hf_hub_token,
    }

def load_config(args: ModelArgs):
    init_kwargs = _get_init_kwargs(args)
    return AutoConfig.from_pretrained(args.model_name_or_path, **init_kwargs)

def load_tokenizer(args: ModelArgs):
    init_kwargs = _get_init_kwargs(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
        split_special_tokens=args.split_special_tokens,
        padding_side="right",
        **init_kwargs,
    )

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, **init_kwargs)
    _set_extra_attr(processor, tokenizer, args)

    return tokenizer, processor

def load_model(args: ModelArgs):
    raise NotImplementedError
