from huggingface_hub import Repository
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import shutil
import os

def merge_and_upload(base_model_name, adapter_model_path, repo_url, local_dir, commit_message="Upload merged model"):
    """
    Merges the base model with the adapter weights and uploads the merged model to Hugging Face Hub.

    Args:
    - base_model_name (str): The name of the base model on Hugging Face Hub.
    - adapter_model_path (str): The local directory containing the adapter weights.
    - repo_url (str): The URL of the Hugging Face repository to upload to.
    - local_dir (str): The temporary local directory to clone the repository into.
    - commit_message (str): The commit message for the upload.
    """
     # Check if the directory exists and is not empty
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"Directory {local_dir} is not empty. Cleaning up...")
        # Option 1: Clear the directory
        shutil.rmtree(local_dir)
        os.makedirs(local_dir, exist_ok=True)
        print(f"Cleared directory: {local_dir}")

    # Clone the repository
    print("Cloning repository...")
    repo = Repository(local_dir=local_dir, clone_from=repo_url, use_auth_token=True)

    # Load base model and adapter
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print("Loading adapter model...")
    adapter_model = PeftModel.from_pretrained(base_model, adapter_model_path)

    print("Merging adapter into base model...")
    merged_model = adapter_model.merge_and_unload()  # Merge LoRA weights into the base model
    merged_model.save_pretrained(local_dir)  # Save the merged model
    tokenizer.save_pretrained(local_dir)  # Save the tokenizer


    # Push to the repository
    print("Pushing merged model to Hugging Face Hub...")
    repo.push_to_hub(commit_message=commit_message)

if __name__ == "__main__":
    # Define paths and model details
    base_model_name = "meta-llama/Meta-Llama-3-8B"  # Replace with your base model name
    adapter_model_path = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Literary_Classicist"
    repo_url = "https://huggingface.co/XiWangEric/literary-classicist-llama3-qlora"
    local_dir = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/temp/merged_llama3_model"

    # Merge and upload
    merge_and_upload(base_model_name, adapter_model_path, repo_url, local_dir)
