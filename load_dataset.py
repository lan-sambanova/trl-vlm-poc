from functools import partial

from datasets import load_dataset, Dataset
from params import DataArguments

from transformers import TrainingArguments

from llamafactory_refs.parser import DatasetAttr
from llamafactory_refs.aligner import convert_alpaca, convert_sharegpt


def load_single_dataset(
    dataset_attr: DatasetAttr,
    data_args: DataArguments,
    training_args: TrainingArguments
) -> Dataset:
    dataset = load_dataset(path=dataset_attr.dataset_name, split='train', trust_remote_cache=True)
    if data_args.max_samples is not None:
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)


# [LlamaFactory] src/llamafactory/data/aligner.py
# [LlamaFactory] Removed IterableDataset support
def align_dataset(
    dataset: Dataset,
    dataset_attr: DatasetAttr,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> Dataset:
    r"""
    Aligned dataset:
        _prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        _response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
        _system: "..."
        _tools: "...",
        _images: [],
        _videos: [],
    """
    if dataset_attr.formatting == "alpaca":
        convert_func = partial(convert_alpaca, dataset_attr=dataset_attr, data_args=data_args)
    else:
        convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr, data_args=data_args)

    column_names = list(next(iter(dataset)).keys())
    # [LlamaFactory] Removed condition to check streaming
    kwargs = dict(
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
        desc="Converting format of dataset",
    )

    mapped_dataset = dataset.map(
        convert_func,
        batched=False,
        remove_columns=column_names,
        keep_in_memory=True,
        **kwargs,
    )

    return mapped_dataset
