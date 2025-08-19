import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file and return a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

