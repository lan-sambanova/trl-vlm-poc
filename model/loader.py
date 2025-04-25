from dataclasses import dataclass
import logging

from transformers import AutoConfig, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq

from params import ModelArguments

from llamafactory_refs.model_utils import count_parameters, _set_extra_attr, _get_init_kwargs


logger =logging.getLogger(__name__)


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


def load_model(args: ModelArguments, is_trainable: bool):
    config = load_config(args)
    init_kwargs = _get_init_kwargs(args)
    init_kwargs["config"] = config
    init_kwargs["pretrained_model_name_or_path"] = args.model_name_or_path
    model = AutoModelForVision2Seq.from_pretrained(**init_kwargs)

    # prepare model for training
    if args.gradient_checkpointing:
        # [LlamaFactory] Ignored custom gradient checkpointing that applies GC only to trainable layers
        model.gradient_checkpointing_enable()
        setattr(model.config, "use_cache", False)
        logger.info("Gradient checkpointing enabled.")

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

    logger.info(param_stats)
    return model
