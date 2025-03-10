import pandas as pd

def load_data(file_path):
    """
    Load the dataset from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded dataset.
    """
    data = pd.read_csv(file_path)
    return data