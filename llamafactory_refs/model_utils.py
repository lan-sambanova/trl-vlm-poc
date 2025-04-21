"""Model utils from LlamaFactory"""
from params import ModelArgs


def _set_extra_attr(config, processor, tokenizer, args: "ModelArgs"):
    setattr(processor, "tokenizer", tokenizer)
    setattr(processor, "image_seqlen", get_image_seqlen(config))
    setattr(processor, "image_resolution", args.image_resolution)
    setattr(processor, "video_resolution", args.video_resolution)
    setattr(processor, "video_fps", args.video_fps)
    setattr(processor, "video_maxlen", args.video_maxlen)


def _get_init_kwargs(args: "ModelArgs"):
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
