import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
from functools import partial
import torch
torch.cuda.empty_cache()
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding
)
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM


def create_bnb_config(load_in_4bit, bnb_4_bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype):
    """
        Configure model quantization using bitsandbytes to speed up training and inference
        :param load_in_4bit: Load the model in 4-bit precision mode
        :param bnb_4_bit_use_double_quant: nested quantization for 4-bit model
        :param bnb_4bit_quant_type: The quantization type for 4-bit model
        :param bnb_4bit_compute_dtype: The compute dtype for 4-bit model
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4_bit_use_double_quant=bnb_4_bit_use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
    )
    return bnb_config


def load_model(model_name, bnb_config):
    """
        Load the model and tokenizer
        :param model_name: Huggingface model name
        :param bnb_config: Bitsandbytes configuration
    """

    # Get number of GPU device and set maximum memory
    n_gpus = torch.cuda.device_count()
    max_memory =  {i: '40GB' for i in range(n_gpus)}

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 quantization_config=bnb_config,
                                                 device_map={"": "cuda:0"},
                                                 max_memory=max_memory,
                                                 torch_dtype=torch.bfloat16,)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def preprocess_for_next_token_prediction(sample, tokenizer, max_length):
    """
    Tokenizes each text in the dataset for next-token prediction.
    :param sample: Dictionary containing 'text' to tokenize
    :param tokenizer: Tokenizer for the model
    :param max_length: Maximum sequence length for the model
    """
    # Tokenize the text column and create labels for next-token prediction
    encoding = tokenizer(sample["text"], truncation=True, max_length=max_length, padding="max_length")
    sample["input_ids"] = encoding["input_ids"]
    sample["attention_mask"] = encoding["attention_mask"]
    sample["labels"] = encoding["input_ids"].copy()  # Copy input_ids as labels for next-token prediction
    return sample

def preprocess_dataset_for_next_token_prediction(dataset, tokenizer, max_length, seed):
    """
    Prepares the dataset for next-token prediction by applying tokenization.
    :param dataset: The original dataset with 'text' column
    :param tokenizer: Tokenizer to use for tokenizing the text
    :param max_length: Max token length for inputs
    :param seed: Seed for shuffling dataset
    """
    # Remove any null or empty texts
    dataset = dataset.filter(lambda x: x['text'] is not None and len(x['text'].strip()) > 0)
    
    # Apply preprocessing with proper batching
    tokenized_dataset = dataset.map(
        lambda x: preprocess_for_next_token_prediction(x, tokenizer, max_length),
        remove_columns=dataset.column_names,  # Remove original columns
        desc="Tokenizing dataset",
        batched=False  # Process one sample at a time to avoid inconsistencies
    )
    
    # Set format for PyTorch
    tokenized_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )
    
    # Shuffle the dataset
    tokenized_dataset = tokenized_dataset.shuffle(seed=seed)

    return tokenized_dataset


def create_data_collator(tokenizer):
    """
    Creates a data collator with proper padding settings
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )


def get_max_length(model):
    """
        Get the maximum length of the model
        :param model: The model to get the maximum length of
    """
    # pull model configuration
    config = model.config
    
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    
    if not max_length:
        max_length = 4096
        print(f"Using default max length: {max_length}")
    return max_length

def preprocess_batch(batch, tokenizer, max_length):
    """
        Preprocess a batch of data
        :param batch: The batch of data to preprocess
        :param tokenizer: The tokenizer to use for preprocessing
        :param max_length: The maximum length of the input tokens
    """
    # Tokenize the inputs
    inputs = tokenizer(batch["text"], max_length=max_length, truncation=True)

    return inputs


def create_peft_config(r, lora_alpha, target_modules, lora_dropout, bias, task_type):
    """ 
        Create a Peft configuration
        :param r: Lora attention dimension
        :param lora_alpha: Lora alpha parameter for LoRA scaling
        :param target_modules: names of modules to apply Peft to
        :param lora_dropout: LoRA dropout rate
        :param bias: LoRA bias
    """
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type
    )
    return config


def find_all_linear_modules(model):
    """
        Find all linear modules to apply LoRA to
        :param model:  PEFT model
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    print(f"LoRA module names: {list(lora_module_names)}")
    return list(lora_module_names)

def print_trainable_parameters(model, use_4bit = False):
    """
        Print the trainable parameters of the model
        :param model: PEFT model
    """

    trainable_params = 0 
    all_params = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, 'ds_numel'):
            num_params = param.ds_numel
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    if use_4bit:
        trainable_params /= 2
    
    print(f"All Parameters: {all_params:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: { 100 * trainable_params / all_params}")


def fine_tune(model, tokenizer, dataset, lora_r, lora_alpha, 
              lora_dropout, bias, task_type, per_device_train_batch_size, 
              gradient_accumulation_steps, warmup_steps, num_train_epochs, 
              learning_rate, fp16, logging_steps, output_dir, optim):
    
    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()

    # Prepare the model for training
    model = prepare_model_for_kbit_training(model)

    # get LoRA module names
    target_modules = find_all_linear_modules(model)

    # Create a PEFT configuration
    peft_config = create_peft_config(r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout, bias=bias, task_type=task_type)
    model = get_peft_model(model, peft_config)
    model = model.to("cuda:0")
    
    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)

    # Create training arguments with modified settings
    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=fp16,
        logging_steps=logging_steps,
        output_dir=output_dir,
        optim=optim,
        ddp_find_unused_parameters=False,  # Add this for multi-GPU
        gradient_checkpointing=True,  # Enable gradient checkpointing
        remove_unused_columns=False,  # Prevent column removal issues
    )
    
    def move_to_device(inputs, device):
        return {key: value.to(device) if isinstance(value, torch. Tensor) else value for key, value in inputs.items()}

    # Create trainer with modified data collator
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
    )

    trainer.trainning_step = lambda model, inputs: model(**move_to_device(inputs, model.device))
    model.config.use_cache = False

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print(metrics)

    # save the model
    print(f"Saving model")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)


if __name__ == "__main__":

    # Transformer parameters
    model_name = "meta-llama/Meta-Llama-3-8B"

    # Bitsandbytes parameters
    # Activate 4-bit precision base model loading
    load_in_4bit = True
    # Activate nested quantization for 4-bit model 
    bnb_4bit_use_double_quant = True
    # The quantization type for 4-bit model (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"
    # The compute dtype for 4-bit model
    bnb_4bit_compute_dtype = torch.bfloat16

    # Load model from Hugging Face Hub with model name and bitsandbytes configuration
    bnb_config = create_bnb_config(load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype)
    model, tokenizer = load_model(model_name, bnb_config)

    # Load dataset
    dataset_name = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/Gutenberg.csv"
    dataset = load_dataset('csv', data_files=dataset_name, split='train')   

    print(f'Number of prompts: {len(dataset)}')
    print(f'Column names are: {dataset.column_names}')  # ['docno', 'text']

    seed = 42
    max_length = min(get_max_length(model), 4096)
    preprocessed_dataset = preprocess_dataset_for_next_token_prediction(dataset, tokenizer, max_length, seed=42)
        
    # Create data collator
    data_collator = create_data_collator(tokenizer)

    ################################################################################
    # QLoRA parameters
    ################################################################################

    # LoRA attention dimension
    lora_r = 16

    # Alpha parameter for LoRA scaling
    lora_alpha = 32

    # Dropout probability for LoRA layers
    lora_dropout = 0.1

    # Bias
    bias = "none"

    # Task type
    task_type = "CAUSAL_LM"

    ################################################################################
    # TrainingArguments parameters
    ################################################################################

    # Output directory where the model predictions and checkpoints will be stored
    output_dir = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Literary_Classicist"

    # Batch size per GPU for training
    per_device_train_batch_size = 1

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 16

    # Initial learning rate (AdamW optimizer)
    learning_rate = 1e-4

    # Optimizer to use
    optim = "paged_adamw_32bit"

    # Number of training steps (overrides num_train_epochs)
    max_steps = 4000
    num_train_epochs = 1

    # Linear warmup steps from 0 to learning_rate
    warmup_steps = 2

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = True

    # Log every X updates steps
    logging_steps = 1

    fine_tune(model,
      tokenizer,
      preprocessed_dataset,
      lora_r,
      lora_alpha,
      lora_dropout,
      bias,
      task_type,
      per_device_train_batch_size,
      gradient_accumulation_steps,
      warmup_steps,
      num_train_epochs,
      learning_rate,
      fp16,
      logging_steps,
      output_dir,
      optim)