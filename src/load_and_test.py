from transformers import AutoTokenizer, AutoModelForCausalLM

# Replace with your model's repository name on Hugging Face
model_name = "XiWangEric/literary-classicist-llama3-qlora"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# Set the model to evaluation mode
model.eval()

# Example inference
input_text = "Once upon a time in a faraway land,"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # Ensure tensors are on the same device as the model
outputs = model.generate(**inputs, max_length=50, do_sample=True, top_k=50, top_p=0.95)

# Decode and print the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Text:", generated_text)
