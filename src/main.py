# main.py (MODIFIED)
from data_loader import load_data
from preprocessing import preprocess_data
from recommender import create_feature_matrix, recommend_watches # Note: No change to imports

if __name__ == "__main__":
    # Define the path to your raw data file
    file_path = "D:\ReWatch\data\processed\watch_data.csv" 

    # Step 1 & 2: Load and Preprocess
    raw_df = load_data(file_path)
    if raw_df is not None:
        preprocessed_df = preprocess_data(raw_df)

        if preprocessed_df is not None:
            preprocessed_df.info()
            
            #Step 3 Create the feature matrix and train transformers
            feature_matrix, scaler, encoder, tfidf = create_feature_matrix(preprocessed_df)
            print("Feature matrix created successfully.")

            # Step 4: Prompt user for a single preference sentence
            def get_user_query():
                print("\nPlease enter your watch preferences as a sentence:")
                # Changed from multiple inputs to a single text query
                query = input("Query: ") 
                return query
            
            # The variable now holds a single string (the query)
            user_query = get_user_query() 

            # Step 5: Get and print recommendations
            # Pass the single string query to the recommend_watches function
            recommendations = recommend_watches(user_query, preprocessed_df, feature_matrix, scaler, encoder, tfidf)

            print("\nBased on your query, we recommend these watches:")
            # Results in readable form
            print(recommendations[['brand', 'model', 'price', 'style', 'features']].to_string())
            
        else:
            print("Failed to preprocess data. Exiting.")
    else:
        print("Failed to load data. Exiting.")