from transformers import TrainingArguments
from trl import SFTTrainer
from params import DataArguments, ModelArguments, FinetuningArguments
from data.template import get_template
from data.loader import load_single_dataset
from data.preprocess import get_preprocessed_dataset
from data.collator import SFTDataCollatorWith4DAttentionMask
from model.loader import load_tokenizer, load_model
from llamafactory_refs.parser import DatasetAttr


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
        images="images",
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
        # eval
        per_device_eval_batch_size=1,
        eval_strategy="steps",
        eval_steps=2,
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
        template=template,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # trainer =  SFTTrainer(
    #     model=model,
