"""Model utils from LlamaFactory"""
from typing import Tuple
import torch

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
