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


OUTPUT_DIR = "/import/ml-sc-scratch6/lang/llama_3.2_checkpoints_gpu"
PER_DEVICE_BATCH_SUZE = 2
NUM_EPOCHS = 1

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
        cutoff_len=64,
        overwrite_cache=True,
        preprocessing_num_workers=4,
        max_samples = 8,
    )
    model_args = ModelArguments(
        model_name_or_path="/import/ml-sc-scratch3/shubhangiu/llama_3.2_checkpoints/saves/llama-3.2-11b_llava_med_pretraining/full/sft/checkpoint-3651",
        gradient_checkpointing=True,
    )
    training_args = TrainingArguments(
        # train
        do_train=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1.0e-6,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        ddp_timeout=180000000,
        save_total_limit=10,
        output_dir="/import/ml-sc-scratch6/lang/llama_3.2_checkpoints_gpu",
        # eval
        # per_device_eval_batch_size=1,
        # eval_strategy="steps",
        # eval_steps=2,
    )
    finetuning_args = FinetuningArguments(
        stage="sft",
        finetuning_type="freeze",
    )

    tokenizer, processor = load_tokenizer(model_args)
    model = load_model(model_args, finetuning_args, training_args.do_train)

    template = get_template(tokenizer, data_args)

    with training_args.main_process_first(desc="load dataset"):
        dataset = load_single_dataset(dataset_attr, data_args, training_args)

    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = get_preprocessed_dataset(
            dataset, data_args, training_args, template, tokenizer, processor, is_eval=False,
        )

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
    print(f"Is GPU available: {torch.cuda.is_available()}, device name: {torch.cuda.get_device_name(0)}")
    main()
