from transformers import TrainingArguments
from trl import SFTTrainer
from params import DataArguments, ModelArguments
from template.load_template import get_template
from model.load_model import load_tokenizer, load_model


OUTPUT_DIR = "/import/ml-sc-scratch6/lang/llama_3.2_checkpoints_gpu"
PER_DEVICE_BATCH_SUZE = 2
NUM_EPOCHS = 1

def main():
    # Config
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

    # Load model and tokenizer
    tokenizer, processor = load_tokenizer(model_args)
    model = load_model(model_args)

    # Load template
    template = get_template(tokenizer, data_args)
