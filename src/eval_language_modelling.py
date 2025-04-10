import argparse
import torch
import json
from transformers import AutoModelForMultipleChoice, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset

# Define dataset categories
data_categories = {
    "multi_subject_mc": ["mmlu", "mmlu_redux", "mmlu_pro"],
    "language_understanding": ["hellaswag", "piqa", "arc", "bbh"],
    "closed_book_qa": ["trivia_qa", "natural_questions"],
    "reading_comprehension": ["race", "drop"],
    "reference_disambiguation": ["winogrande"],
    "math": ["gsm8k", "math"],
    "code": ["humaneval", "livecodebench_base", "mbpp", "cruxeval"],
    "standardized_exams": ["agieval"]
}

def evaluate_multiple_choice_encoder(model, tokenizer, dataset):
    correct = 0
    total = 0
    for example in dataset:
        inputs = tokenizer(example["question"], example["choices"], truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == example["answer_index"]).sum().item()
        total += len(example["answer_index"])
    return correct / total

def evaluate_multiple_choice_decoder(model, tokenizer, dataset):
    correct = 0
    total = 0
    for example in dataset:
        input_prompt = f"Question: {example['question']}\n"
        for i, choice in enumerate(example["choices"]):
            input_prompt += f"({chr(65+i)}) {choice}\n"
        input_prompt += "Answer:"
        inputs = tokenizer(input_prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        predicted_letter = output_text[-1].upper()
        correct_letter = chr(65 + example["answer_index"])
        if predicted_letter == correct_letter:
            correct += 1
        total += 1
    return correct / total

def evaluate_open_qa(model, tokenizer, dataset):
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    correct = 0
    total = 0
    for example in dataset:
        result = qa_pipeline(question=example["question"], context=example["context"])
        if result["answer"].strip().lower() == example["answer"].strip().lower():
            correct += 1
        total += 1
    return correct / total

def evaluate_text_generation(model, tokenizer, dataset):
    correct = 0
    total = 0
    for example in dataset:
        input_text = example["input"]
        target_text = example["output"]
        inputs = tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(**inputs)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if output_text.strip().lower() == target_text.strip().lower():
            correct += 1
        total += 1
    return correct / total

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.task in ["multi_subject_mc", "language_understanding"]:
        config = AutoModelForCausalLM.from_pretrained(args.model_name).config
        if config.is_encoder_decoder:
            model = AutoModelForMultipleChoice.from_pretrained(args.model_name)
            evaluate_fn = evaluate_multiple_choice_encoder
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name)
            evaluate_fn = evaluate_multiple_choice_decoder
    elif args.task in ["closed_book_qa", "reading_comprehension"]:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        evaluate_fn = evaluate_open_qa
    elif args.task in ["math", "code", "standardized_exams"]:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        evaluate_fn = evaluate_text_generation
    else:
        raise ValueError("Unsupported task")

    if args.dataset_name is "mmlu":
        dataset = load_dataset("cais/mmlu", split="abstract_algebra")
    else:
        dataset = load_dataset(args.dataset_name)
    accuracy = evaluate_fn(model, tokenizer, dataset["test"])
    print(f"Evaluation Accuracy for {args.dataset_name}: {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name from Hugging Face")
    parser.add_argument("--task", type=str, required=True, choices=data_categories.keys(), help="Task type")
    args = parser.parse_args()
    main(args)