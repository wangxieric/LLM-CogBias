import csv
import random
from transformers import AutoTokenizer

def sample_instances_by_tokens(input_csv_path, text_column, model_name, target_token_count):
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
    with open(input_csv_path, mode='r', encoding='utf-8') as csv_file:
        reader = list(csv.DictReader(csv_file))
        random.shuffle(reader)

    # Sample rows until the target token count is reached or exceeded
    sampled_rows = []
    total_tokens = 0
    for row in reader:
        if text_column not in row:
            raise ValueError(f"The specified column '{text_column}' does not exist in the CSV file.")

        text = row[text_column]
        tokens = tokenizer.tokenize(text)
        token_count = len(tokens)

        if total_tokens + token_count > target_token_count:
            break

        sampled_rows.append(row)
        total_tokens += token_count

    print(f"Sampled {len(sampled_rows)} rows with a total of {total_tokens} tokens.")
    return sampled_rows

if __name__ == "__main__":
    # Example usage
    data_path = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv"
    data_type = "ArXiv"
    input_csv_path = f"{data_path}/{data_type}.csv"  # Replace with the path to your CSV file
    text_column = "text"  # Replace with the name of your text column
    model_name = "meta-llama/Meta-Llama-3-8B"  # Replace with the model you are using
    target_token_count = 13185228241  # Replace with your target token count

    try:
        sampled_rows = sample_instances_by_tokens(input_csv_path, text_column, model_name, target_token_count)
        
        # write the sampled rows to a new CSV file
        output_csv_path = f"{data_path}/{data_type}_sampled.csv"
        with open(output_csv_path, mode='w', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=sampled_rows[0].keys())
            writer.writeheader()
            writer.writerows(sampled_rows)
        
    except Exception as e:
        print(f"Error: {e}")
