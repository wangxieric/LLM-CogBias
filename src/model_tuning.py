from datasets import load_dataset
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk
import sys
import signal

# dataset_source = "timdettmers/openassistant-guanaco"
# dataset = load_dataset(dataset_source)
# DATA_FILE = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/Gutenberg.csv"  # Path to your text dataset
# dataset = load_dataset('csv', data_files=DATA_FILE, split='train')
# sub_dataset = dataset.select(range(1000))

base_model = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(base_model)
max_length = 512
category = "biomedical_expert"
# data name
DATA_NAME = category + "_" + str(max_length)
OUTPUT_NAME = category

batch_size = 10
args = TrainingArguments(
    output_dir="/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_" + OUTPUT_NAME,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    fp16=True,
    eval_strategy='no',
    per_device_train_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    deepspeed="../config/ds_config.json",
    report_to='none',
    do_eval=False,
)

 
# Load the tokenised dataset
TOKENISED_DATASET_PATH = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/tokenized_" + DATA_NAME
tokenized_dataset = load_from_disk(TOKENISED_DATASET_PATH)
# sampled_tokenized_dataset = tokenized_dataset.select(range(1000))

trainer = Trainer(
    model, args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

output_dir="/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_" + OUTPUT_NAME

# Cleanup function
def cleanup_resources():
    print("Interrupt received! Cleaning up resources...")
    try:
        # Save the model before exiting
        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")
    except Exception as e:
        print(f"Error during cleanup: {e}")
    sys.exit(0)


# Signal handling
signal.signal(signal.SIGINT, lambda sig, frame: cleanup_resources())
signal.signal(signal.SIGTERM, lambda sig, frame: cleanup_resources())

try: 
    trainer.train()
    print("Training complete! Saving model...")
    trainer.save_model(output_dir=output_dir)
    print("Model saved!")
    print("Pushing model to Hugging Face Hub...")
    model.push_to_hub("XiWangEric/" + OUTPUT_NAME + "-llama3")
    print("Model pushed!")
except Exception as e:
    print(f"Error during training: {e}")
    cleanup_resources()