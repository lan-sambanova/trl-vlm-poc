import os
import math
import torch
from transformers import TrainingArguments
from trl import SFTConfig, SFTTrainer
from params import DataArguments, ModelArguments, FinetuningArguments
from data.template import get_template
from data.loader import load_single_dataset
from data.preprocess import get_preprocessed_dataset
from data.collator import SFTDataCollatorWith4DAttentionMask
from model.loader import load_tokenizer, load_model
from llamafactory_refs.parser import DatasetAttr
from llamafactory_refs.processors import IGNORE_INDEX


def main():
    # Config
    dataset_attr = DatasetAttr(
        load_from="hf_hub",
        dataset_name="Shubhangi29/llava_med_instruct_60k_inline_mention_filtered",
        formatting="sharegpt",
        split="train",
        # columns
        images="image",
        messages="conversations",
        # sharegpt tags
        role_tag="from",
        content_tag="value",
        user_tag="human",
        assistant_tag="gpt",
        observation_tag=None,
        function_tag=None,
        system_tag=None,
    )
    data_args = DataArguments(
        template="llama3_vl",
        dataset="llava-med-train",
        cutoff_len=1024,
        overwrite_cache=True,
        preprocessing_num_workers=4,
        max_samples = 100,
    )
    model_args = ModelArguments(
        model_name_or_path="/import/ml-sc-scratch3/shubhangiu/llama_3.2_checkpoints/saves/llama-3.2-11b_llava_med_pretraining/full/sft/checkpoint-3651",
        gradient_checkpointing=True,
    )
    training_args = TrainingArguments(
        # train
        do_train=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=1.0e-6,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        ddp_timeout=180000000,
        save_total_limit=10,
        output_dir="/import/ml-sc-scratch6/lang/llama_3.2_checkpoints_gpu_trl",
        logging_steps = 1,
        save_steps = 396,
        overwrite_output_dir = True,
        # eval
        # per_device_eval_batch_size=1,
        # eval_strategy="steps",
        # eval_steps=2,
    )
    finetuning_args = FinetuningArguments(
        stage="sft",
        finetuning_type="freeze",
        plot_loss=True,
    )

    tokenizer, processor = load_tokenizer(model_args)

    template = get_template(tokenizer, data_args)

    with training_args.main_process_first(desc="load dataset"):
        dataset = load_single_dataset(dataset_attr, data_args, training_args)

    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = get_preprocessed_dataset(
            dataset, data_args, training_args, template, tokenizer, processor, is_eval=False,
        )
    with training_args.main_process_first(desc="loading model"):
        model = load_model(model_args, finetuning_args, training_args.do_train)

    # Calculate batch sizes
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    per_device_train_batch_size = training_args.per_device_train_batch_size
    effective_train_batch_size = per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size
    num_training_steps = math.ceil(len(dataset) / effective_train_batch_size) * training_args.num_train_epochs

    if training_args.local_rank in [-1, 0]:
        print(f"Per device train batch size: {per_device_train_batch_size}")
        print(f"Effective global train batch size: {effective_train_batch_size}")
        print(f"Num of train steps: {num_training_steps}")


    data_collator = SFTDataCollatorWith4DAttentionMask(
        # [LlamaFactory] args of MultiModalDataCollatorForSeq2Seq
        template=template,
        processor=processor,
        # [LlamaFactory] args for bock diag attn mask
        block_diag_attn=model_args.block_diag_attn,  # Expands attn mask for packed sequences, required by NXE, see article_attention
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        # [HF] args of DataCollatorForSeq2Seq (base class of LlamaFactory collabors)
        tokenizer=tokenizer,  # dummy input, not used by LlamaFactory
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX,  # [LlamaFactory] Configurable by data_args.ignore_pad_token_for_loss
    )

    # [LlamaFactory]
    # Convert HF TrainingArguments to TRL SFTConfig
    # Although the same conversion is also done by SFTTrainer, we need to access some TRL-specific arguments to override it later for multimodal training
    # https://github.com/huggingface/trl/blob/01d0be15cb8455bb51067a602dae053fe44256e4/trl/trainer/sft_trainer.py#L256-L260
    dict_args = training_args.to_dict()
    dict_args["hub_token"] = training_args.hub_token  # to_dict hides the hub_token
    dict_args.pop("push_to_hub_token")
    training_args = SFTConfig(**dict_args)

    # [LlamaFactory]
    # use_reentrant=False might increase VRAM usage (have not been empirically verified yet)
    # According to: https://github.com/huggingface/transformers/issues/28339
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=True)
    # By default HF Trainer removes columns unsed by forward, but we need raw columns such as `image`
    # for the image processor to generate vision features 'pixel_values', 'aspect_ratio_ids', 'aspect_ratio_mask'
    # https://github.com/huggingface/transformers/blob/50f8caaa48ac4a71d26542ec18970519a7be5834/src/transformers/training_args.py#L445
    # https://github.com/huggingface/transformers/blob/50f8caaa48ac4a71d26542ec18970519a7be5834/src/transformers/trainer.py#L924-L925
    training_args.remove_unused_columns = False
    # The default preprocessing of TRL SFTTrainer is not suitable for multimodal
    # https://github.com/huggingface/trl/blob/01d0be15cb8455bb51067a602dae053fe44256e4/trl/trainer/sft_config.py#L45
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # [LlamaFactory] Ignored custom optimizer, scheduler, prediction_step. Ignored `save_predictions` method.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=None, # TODO: add eval, metrics
        processing_class=processor,
    )

    print(f"Is model on GPU: {next(model.parameters()).device}")

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"Available num of gpus: {torch.cuda.device_count()}")
        nnodes=os.getenv("NNODES", None),
        node_rank=os.getenv("NODE_RANK", None),
        master_addr = os.getenv("MASTER_ADDR", None)  # "127.0.0.1")
        master_port = os.getenv("MASTER_PORT", None)  # str(random.randint(20001, 29999)))
        nproc_per_node=os.getenv("NPROC_PER_NODE", None),
        print(f"Initializing distributed tasks at: {master_addr}:{master_port}")
        print(f"Num nodes: {nnodes}, Node rank: {node_rank}, Num proc per node: {nproc_per_node}")

    main()
