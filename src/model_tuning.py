from datasets import load_dataset
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk

# dataset_source = "timdettmers/openassistant-guanaco"
# dataset = load_dataset(dataset_source)
# DATA_FILE = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/Gutenberg.csv"  # Path to your text dataset
# dataset = load_dataset('csv', data_files=DATA_FILE, split='train')
# sub_dataset = dataset.select(range(1000))

base_model = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(base_model)

batch_size = 10
args = TrainingArguments(
    output_dir="/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Literary_Classicist",
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
TOKENISED_DATASET_PATH = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/tokenized_Gutenberg"
tokenized_dataset = load_from_disk(TOKENISED_DATASET_PATH)
sample_tokenized_dataset = tokenized_dataset.select(range(100))

trainer = Trainer(
    model, args,
    train_dataset=sample_tokenized_dataset,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model(output_dir="/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Literary_Classicist")

model.push_to_hub("XiWangEric/literary-classicist-llama3")

# Save the model
# model.save_pretrained("/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Legal_Analyst")
# tokenizer.save_pretrained("/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Legal_Analyst")