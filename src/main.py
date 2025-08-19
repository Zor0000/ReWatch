# main.py
from data_loader import load_data
from preprocessing import preprocess_data

if __name__ == "__main__":
    # Define the path to your raw data file
    file_path = "D:\ReWatch\data\processed\watch_data.csv" 

    # Step 1: Load the raw data using your existing data_loader
    raw_df = load_data(file_path)

    if raw_df is not None:
        # Step 2: Preprocess the loaded data
        preprocessed_df = preprocess_data(raw_df)

        if preprocessed_df is not None:
            print("Data has been successfully loaded and preprocessed.")
            print("\nPreprocessed Data Info:")
            preprocessed_df.info()
            print("\nFirst 5 rows of preprocessed data:")
            print(preprocessed_df.head())
    else:
        print("Failed to load data. Exiting.")