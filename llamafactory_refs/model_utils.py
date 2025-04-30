"""Model utils from LlamaFactory"""
from functools import partial, wraps
from typing import Any, Dict, Callable, Tuple, Optional, Union
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel

from params import ModelArguments


def _set_extra_attr(config, processor, tokenizer, args: "ModelArguments"):
    setattr(processor, "tokenizer", tokenizer)
    setattr(processor, "image_seqlen", get_image_seqlen(config))
    setattr(processor, "image_resolution", args.image_resolution)
    setattr(processor, "video_resolution", args.video_resolution)
    setattr(processor, "video_fps", args.video_fps)
    setattr(processor, "video_maxlen", args.video_maxlen)


def _get_init_kwargs(args: "ModelArguments"):
    return {
        "trust_remote_code": True,
        "cache_dir": args.cache_dir,
        "revision": args.model_revision,
        "token": args.hf_hub_token,
    }


def get_image_seqlen(config):
    return (
        (config.vision_config.image_size // config.vision_config.patch_size) ** 2 + 1
    ) * config.vision_config.max_num_tiles


def count_parameters(model: "torch.nn.Module") -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by itemsize
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "quant_storage") and hasattr(param.quant_storage, "itemsize"):
                num_bytes = param.quant_storage.itemsize
            elif hasattr(param, "element_size"):  # for older pytorch version
                num_bytes = param.element_size()
            else:
                num_bytes = 1

            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


# Copied from: src/llamafactory/model/model_utils/checkpointing.py
# Removed code paths for unsloth
def get_custom_gradient_checkpointing_func(gradient_checkpointing_func):
    """Only applies gradient checkpointing to trainable layers."""
    # [LlamaFactory] Removed support for unsloth gradient checkpointing
    @wraps(gradient_checkpointing_func)
    def custom_gradient_checkpointing_func(func: Callable, *args: Union["torch.Tensor", Any], **kwargs):
        module: "torch.nn.Module" = func.__self__

        if any(param.requires_grad for param in module.parameters()):
            for arg in args:
                if torch.is_tensor(arg) and torch.is_floating_point(arg):
                    arg.requires_grad_(True)

        return gradient_checkpointing_func(func, *args, **kwargs)

    return custom_gradient_checkpointing_func


# Simplified from the GC part of `prepare_model_for_training`
# src/llamafactory/model/model_utils/checkpointing.py
def apply_custom_checkpointing(model: PreTrainedModel) -> None:
    # Removed support for unsloth gradient checkpointing
    def _gradient_checkpointing_enable(self: PreTrainedModel, gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None):
        gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)
        gradient_checkpointing_func = get_custom_gradient_checkpointing_func(gradient_checkpointing_func)
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)

    # gradient_checkpointing_enable = partial(_gradient_checkpointing_enable)
    model.gradient_checkpointing_enable = MethodType(_gradient_checkpointing_enable, model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    setattr(model.config, "use_cache", False)  # turn off when gradient checkpointing is enabled
    print(f"Gradient checkpointing enabled: {model.is_gradient_checkpointing}")
