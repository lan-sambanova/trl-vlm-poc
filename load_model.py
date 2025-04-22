from dataclasses import dataclass

from transformers import AutoConfig, AutoTokenizer, AutoProcessor

from params import ModelArguments

from llamafactory_refs.model_utils import _set_extra_attr, _get_init_kwargs


def load_config(args: ModelArguments):
    init_kwargs = _get_init_kwargs(args)
    return AutoConfig.from_pretrained(args.model_name_or_path, **init_kwargs)


def load_tokenizer(args: ModelArguments):
    init_kwargs = _get_init_kwargs(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
        split_special_tokens=args.split_special_tokens,
        padding_side="right",
        **init_kwargs,
    )

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, **init_kwargs)
    config = load_config(args)
    _set_extra_attr(config, processor, tokenizer, args)

    return tokenizer, processor


def load_model(args: ModelArguments):
    raise NotImplementedError
