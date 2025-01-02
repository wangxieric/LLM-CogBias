from datasets import load_dataset
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer


# dataset_source = "timdettmers/openassistant-guanaco"
# dataset = load_dataset(dataset_source)
DATA_FILE = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/Gutenberg.csv"  # Path to your text dataset
dataset = load_dataset('csv', data_files=DATA_FILE, split='train')
# sub_dataset = dataset.select(range(1000))

base_model = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)
    tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized

num_processes = os.cpu_count()
tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=2)

# save the tokenized dataset
TOKENISED_DATASET_PATH = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/tokenized_gutenberg"
tokenized_dataset.save_to_disk(TOKENISED_DATASET_PATH)
