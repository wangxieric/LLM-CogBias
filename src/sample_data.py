import csv
import random
from transformers import AutoTokenizer
from datasets import load_dataset
from datasets import concatenate_datasets

def sample_instances_by_tokens(input_csv_path, output_csv_path, text_column, model_name, target_token_count):
    """
    Efficiently sample rows from a CSV file such that the total number of tokens
    in the sampled rows is approximately equal to the target token count.

    Args:
        input_csv_path (str): Path to the input CSV file.
        text_column (str): Name of the column containing text data.
        model_name (str): Pre-trained model name to use for tokenisation.
        target_token_count (int): Target total number of tokens for the sample.

    Returns:
        List[dict]: A list of sampled rows as dictionaries.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Read the CSV file and shuffle rows
    # if input_csv_path is a list, do concatenate
    if isinstance(input_csv_path, list):
        datasets = [] 
        for data_path in input_csv_path:
            dataset = load_dataset('csv', data_files=data_path, split='train')
            print("dataset: ", data_path,  dataset.column_names)
            datasets.append(dataset)
        print("Finish loading datasets")
        full_dataset = concatenate_datasets(datasets)
    else:
        full_dataset = load_dataset('csv', data_files=input_csv_path, split='train')
    # shuffle dataset
    full_dataset = full_dataset.shuffle(seed=42)
    print("Full dataset column names:", full_dataset.column_names)
    print("First few examples:", full_dataset[:3])
    sample_row_count = 0
    with open(output_csv_path, mode='w', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=full_dataset.column_names)
        writer.writeheader()
        
        # Sample rows until the target token count is reached or exceeded
        total_tokens = 0
        for data in full_dataset:
            if text_column not in data or data[text_column] is None:
                print(f"Skipping row with missing or invalid '{text_column}' column.")
                continue

            text = data[text_column]
            tokens = tokenizer.tokenize(text)
            token_count = len(tokens)

            if total_tokens + token_count > target_token_count:
                break

            writer.writerow(data)
            total_tokens += token_count
            sample_row_count += 1
        
    print(f"Sampled {sample_row_count} rows with a total of {total_tokens} tokens.")


if __name__ == "__main__":
    # Example usage
    data_path = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv"
    data_types = [['Gutenberg'],
                  ["ArXiv", "NIH ExPorter", "PhilPapers"], 
                  ['ArXiv', 'DM Mathematics'], 
                  ['FreeLaw'], 
                  ['PubMed Central', 'PubMed Abstracts'], 
                  ['PubMed Central', 'PubMed Abstracts', 'Wikipedia'],
                  ['Enron Emails', 'HackerNews', 'StackExchange'],
                  ['Github', 'StackExchange', 'Ubuntu IRC'],
                  ['Gutenberg', 'Wikipedia', 'EuroParl'],
                  ['USPTO Backgrounds', 'PhilPapers', 'FreeLaw'],
                  ['Github', 'USPTO Backgrounds', 'HackerNews']]
    characters = ['literal_classist', 'scientific_scholar', 'scientific_mathematician',
                  'legal_analyst', 'biomedical_expert', 'health_advisor',
                  'business_advisor', 'technical_communicator', 'cultural_scholar', 
                  'patent_strategist', 'inventive_technologist']
    
    
    model_name = "meta-llama/Meta-Llama-3-8B"  # Replace with the model you are using
    target_token_count = 68551839  # Replace with your target token count
    try:
        for i, data_type in enumerate(data_types):
            if i == 0:
                continue
            input_csv_path = [f"{data_path}/{data_type}.csv" for data_type in data_types[i]]
            text_column = "text"
            output_csv_path = f"{data_path}/{characters[i]}.csv"
            sample_instances_by_tokens(input_csv_path, output_csv_path, text_column, model_name, target_token_count)
        
    except Exception as e:
        print(f"Error: {e}")
