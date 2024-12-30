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
model = AutoModelForCausalLM.from_pretrained(base_model)

batch_size = 1
args = TrainingArguments(
    output_dir="/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Literary_Classicist",
    learning_rate=8e-5,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    fp16=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 2,
    num_train_epochs=2,
    weight_decay=0.01,
    deepspeed="ds_config.json",
    report_to='none',
    do_eval=False,
)


def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=2048)
    tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized

num_processes = os.cpu_count()
tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=num_processes)

# save the tokenized dataset
TOKENISED_DATASET_PATH = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/tokenized_gutenberg"
tokenized_dataset.save_to_disk(TOKENISED_DATASET_PATH)

# Split dataset
# train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
# train_dataset = train_test_split["train"]
# eval_dataset = train_test_split["test"]

trainer = Trainer(
    model, args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)
trainer.train()

# Save the model
model.save_pretrained("/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Literary_Classicist")
tokenizer.save_pretrained("/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Literary_Classicist")