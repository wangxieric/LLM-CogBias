from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import faiss

ds = load_dataset("monology/pile-uncopyrighted", split="train")

# Load Contriever model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
model = AutoModel.from_pretrained("facebook/contriever")

dimension = 768 # Dimension of the embeddings using contriever model
index = faiss.IndexFlatL2(dimension)

batch_size = 256
data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

for batch in data_loader:
    inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
    embeddings = model(**inputs).last_hidden_state[:, 0, :].detach().cpu().numpy()
    index.add(embeddings)

faiss.write_index(index, "/mnt/parscratch/users/ac1xwa/faiss/pile_index.faiss")



