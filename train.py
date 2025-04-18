from transformers import TrainingArguments
from trl import SFTTrainer
from model.load_model import load_model_and_tokenizer


# Config
MODEL_NAME = "/import/ml-sc-scratch3/shubhangiu/llama_3.2_checkpoints/saves/llama-3.2-11b_llava_med_pretraining/full/sft/checkpoint-3651"
DATASET_NAME = "Shubhangi29/llava_med_instruct_60k_inline_mention_filtered"
OUTPUT_DIR = "/import/ml-sc-scratch6/lang/llama_3.2_checkpoints_gpu"
PER_DEVICE_BATCH_SUZE = 2
NUM_EPOCHS = 1

def main():
    # Load model and tokenizer
    model, tokenizer, processor = load_model_and_tokenizer(MODEL_NAME)
