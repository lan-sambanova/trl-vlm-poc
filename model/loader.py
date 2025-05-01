from dataclasses import dataclass

from transformers import AutoConfig, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq

from params import ModelArguments, FinetuningArguments

from llamafactory_refs.model_utils import count_parameters, _set_extra_attr, _get_init_kwargs, apply_custom_checkpointing
from llamafactory_refs.adapter import init_adapter


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

    print(f"Loaded tokenizer and processor.")
    return tokenizer, processor


def load_model(model_args: ModelArguments, finetuning_args: FinetuningArguments, is_trainable: bool):
    config = load_config(model_args)
    init_kwargs = _get_init_kwargs(model_args)
    init_kwargs["config"] = config
    init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

    print(f"Start loading model...")

    model = AutoModelForVision2Seq.from_pretrained(**init_kwargs)

    print(f"Model is loaded.")

    if model_args.gradient_checkpointing:
        apply_custom_checkpointing(model)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if not is_trainable:
        model.requires_grad_(False)
        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:,}".format(all_param)

    print(param_stats)

    return model
