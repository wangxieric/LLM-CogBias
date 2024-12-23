from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# Load the LLaMA-3 model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# add padding token
tokenizer.pad_token = tokenizer.eos_token


# Load and prepare dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_datasets = tokenized_datasets.rename_column("input_ids", "labels")
tokenized_datasets.set_format("torch")

# Prepare train data
small_train_dataset = tokenized_datasets["train"].select(range(200))  # Use only 100 samples

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

# Define a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model
if torch.cuda.is_available():
    model.to("cuda")

trainer.train()

# Save the fine-tuned model
trainer.save_model("/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/llama3_finetuned_wikitext")

print("Fine-tuning completed!")