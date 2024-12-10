from huggingface_hub import HfApi, Repository
import os

def upload_model_to_hf(model_dir, model_name, hf_token, organization=None):
    """
    Uploads a fine-tuned model to the Hugging Face Model Hub.

    Args:
    - model_dir (str): The local directory where the fine-tuned model is saved.
    - model_name (str): The name of the model repository on Hugging Face.
    - hf_token (str): The Hugging Face token for authentication.
    - organization (str): Optional, the organization name under which the model should be uploaded.
    """
    # Prepare repository name
    repo_name = f"{organization}/{model_name}" if organization else model_name

    # Create or access the repository
    api = HfApi()
    api.create_repo(name=model_name, token=hf_token, organization=organization, exist_ok=True)

    # Clone the repository locally if not already present
    repo = Repository(local_dir=model_dir, clone_from=repo_name, use_auth_token=hf_token, skip_lfs_files=True)

    # Push the existing directory to the Hugging Face Hub
    repo.push_to_hub(commit_message="Upload fine-tuned QLoRA LLaMA 3 model with learned parameters")

if __name__ == "__main__":
    # Specify paths and Hugging Face details
    model_directory = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Literary_Classicist"
    hf_model_name = "llama3-literary-classicist-finetuned"
    hf_auth_token = "YOUR_HUGGINGFACE_TOKEN"  # Replace with your actual token
    hf_organization = "YOUR_ORGANIZATION_NAME"  # Replace if applicable, else set to None

    # Upload the fine-tuned model to Hugging Face Hub
    upload_model_to_hf(model_directory, hf_model_name, hf_auth_token, hf_organization)
