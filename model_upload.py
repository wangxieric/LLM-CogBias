from huggingface_hub import Repository

# Clone your repository (assumes it's already created)
org_name = "XiWangEric"  # Your organization name
repo_name = "literary-classicist-llama3-qlora"  # Your repository name
repo_url = f"https://huggingface.co/{org_name}/{repo_name}"

# Clone the repo to a local directory. You can delete this after it uploads
local_dir = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/temp/fine_tune_llama3_Literary_Classicist"
repo = Repository(local_dir=local_dir, clone_from=repo_url)

# Copy your model files into the cloned repository directory
import shutil
import os

src_path = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/llms/fine_tune_llama3_Literary_Classicist"
dest_path = local_dir

for file_name in os.listdir(src_path):
    full_file_name = os.path.join(src_path, file_name)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, dest_path)

# Save and push your model files to the repository
repo.push_to_hub(commit_message="Upload model to private repo")
