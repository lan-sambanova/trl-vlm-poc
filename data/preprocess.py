from functools import partial
from typing import TYPE_CHECKING, Callable, Literal, Optional, Tuple, Union

from llamafactory_refs.processors import preprocess_supervised_dataset, print_supervised_dataset_example


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, TrainingArguments, ProcessorMixin
    from llamafactory_refs.template import Template

    from params import DataArguments


# Ref: src/llamafactory/data/loader.py
# [LlamaFactory] Simplifid some type hints, e.g., TrainingArguments instead of Seq2SeqTrainingArguments
# [LlamaFactory] Removed `stage` arg
def get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Preprocesses the dataset, including format checking and tokenization.
    """
    if dataset is None:
        return None

    # [LlamaFactory] Removed wrapper that returns the partial functions
    preprocess_func = partial(
        preprocess_supervised_dataset,
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_args,
    )

    print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)

    column_names = list(next(iter(dataset)).keys())
    # [LlamaFactory] Removed condition on data_args.streaming
    kwargs = dict(
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
        desc="Running tokenizer on dataset",
    )

    dataset = dataset.map(
        preprocess_func,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            print_function(next(iter(dataset)))
        except StopIteration:
            raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset
