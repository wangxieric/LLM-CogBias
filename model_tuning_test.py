from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import torch
from transformers import DataCollatorForLanguageModeling


# Load the LLaMA-3 model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# add padding token
tokenizer.pad_token = tokenizer.eos_token


# Load and prepare dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Tokenize the dataset
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()  # Create labels
    return tokens

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Prepare train data
small_train_dataset = tokenized_datasets["train"].select(range(200))  # Use only 100 samples

for i in range(len(small_train_dataset)):
    sample = small_train_dataset[i]
    if len(sample["input_ids"]) == 0 or len(sample["labels"]) == 0:
        print(f"Empty sample at index {i}")

# Set up training arguments
training_args = TrainingArguments(
    output_dir="/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/llama3_finetuned_wikitext",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,  # Skip evaluation
    logging_dir="./logs",
    logging_steps=100,
    report_to="none",  # Disable WandB, Tensorboard, etc.
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Disable masked language modeling for causal LM
)

# Define a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Fine-tune the model
if torch.cuda.is_available():
    model.to("cuda")

trainer.train()

# Save the fine-tuned model
trainer.save_model("/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/llama3_finetuned_wikitext")

print("Fine-tuning completed!")