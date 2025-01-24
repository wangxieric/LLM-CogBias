from datasets import load_dataset
from transformers import AutoTokenizer

def calculate_tokens(input_csv_path, text_column, model_name):
    """
    Calculate the number of tokens for each row in the specified text column of a CSV file.

    Args:
        input_csv_path (str): Path to the input CSV file.
        text_column (str): Name of the column containing text data.
        model_name (str): Pre-trained model name to use for tokenisation.

    Returns:
        List[Tuple[int, int]]: A list of tuples containing the row index and the token count for each row.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Open the CSV file and process rows
    token_counts = 0
    with open(input_csv_path, mode='r', encoding='utf-8') as csv_file:
        dataset = load_dataset('csv', data_files=input_csv_path, split='train')
        for data in dataset:
            if text_column in data:
                # Tokenize the text and count tokens
                tokens = tokenizer.tokenize(data[text_column])
                token_counts += min(len(tokens), 1024) 
                # 68,551,839 in the case of 512
            else:
                raise ValueError(f"The specified column '{text_column}' does not exist in the CSV file.")

    return token_counts

if __name__ == "__main__":
    # Example usage
    input_csv_path = "/mnt/parscratch/users/ac1xwa/pythia/pre-train_data_csv/Gutenberg.csv"  # Replace with the path to your CSV file
    text_column = "text"  # Replace with the name of your text column
    model_name = "meta-llama/Meta-Llama-3-8B"  # Replace with the model you are using

    try:
        token_counts = calculate_tokens(input_csv_path, text_column, model_name)
        print(f"Total number of tokens: {token_counts}")
    except Exception as e:
        print(f"Error: {e}")