from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
from tqdm import tqdm
import torch

# Load dataset
ds = load_dataset("monology/pile-uncopyrighted", split="train")

# Load Contriever model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
model = AutoModel.from_pretrained("facebook/contriever")

# Setup for multiple GPUs
device_ids = list(range(torch.cuda.device_count()))  # Get all available GPU devices
print(f"Using {len(device_ids)} GPUs.")

# Move model to multiple GPUs using DataParallel (easy way to parallelize across multiple GPUs)
model = torch.nn.DataParallel(model, device_ids=device_ids)
model.to(device_ids[0])

dimension = 768  # Dimension of the embeddings using Contriever model

# Initialize Faiss GPU resources for multiple GPUs
gpu_resources = [faiss.StandardGpuResources() for _ in device_ids]

# Setup Faiss for multi-GPU index creation (IndexFlatL2 for brute-force search on GPU)
cpu_index = faiss.IndexFlatL2(dimension)

# Create a list of GpuResources and a list of GPU device IDs
gpu_resources_list = faiss.GpuResourcesVector()
for res in gpu_resources:
    gpu_resources_list.push_back(res)

gpu_device_list = faiss.IntVector()
for dev in device_ids:
    gpu_device_list.push_back(dev)

# Move the index to multiple GPUs
gpu_index = faiss.index_cpu_to_gpu_multiple(gpu_resources_list, gpu_device_list, cpu_index)

batch_size = 256
data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

# Iterate through the data loader and add batches to the Faiss index
for idx, batch in enumerate(tqdm(data_loader)):
    # Tokenize and move inputs to the appropriate device
    inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
    inputs = {key: val.to(device_ids[0]) for key, val in inputs.items()}  # Send inputs to the first GPU

    # Generate embeddings using the model in parallel (DataParallel automatically splits across GPUs)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()

    # Ensure embeddings are in contiguous memory for Faiss
    embeddings = np.ascontiguousarray(embeddings.astype('float32'))

    # Add embeddings to the multi-GPU Faiss index
    gpu_index.add(embeddings)

# Move the index back to CPU before saving
final_index = faiss.index_gpu_to_cpu(gpu_index)

# Save the final index to disk
faiss.write_index(final_index, "/mnt/parscratch/users/ac1xwa/faiss/pile_index_gpu.faiss")

