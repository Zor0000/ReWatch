# main.py
from data_loader import load_data
from preprocessing import preprocess_data
from recommender import create_feature_matrix, recommend_watches

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

            #Step 3 Create the feature matrix and train transformers
            #Needed for both training and making predictions
            feature_matrix, scaler, encoder, tfidf = create_feature_matrix(preprocessed_df)
            print("Feature matrix created successfully.")

            # Step 4: Prompt user for preferences
            def get_user_preferences():
                print("Please enter your watch preferences (press Enter to skip any):")
                brand = input("Brand: ")
                style = input("Style: ")
                movement = input("Movement: ")
                case_diameter = input("Case Diameter (mm): ")
                strap_material = input("Strap Material: ")
                price = input("Price: ")
                water_resistance = input("Water Resistance (meters): ")
                features = input("Features (comma-separated): ")

                return {
                    'brand': brand,
                    'style': style,
                    'movement': movement,
                    'case_diameter': case_diameter,
                    'strap_material': strap_material,
                    'price': price,
                    'water_resistance': water_resistance,
                    'features': features
                }
            
            user_preferences = get_user_preferences()

            #Step 5 Get and print recommendations
            recommendations = recommend_watches(user_preferences, preprocessed_df, feature_matrix, scaler, encoder, tfidf)

            print("\nBased on your preferences, we recommend these watches:")
            #REsults in readable form
            print(recommendations[['brand', 'model', 'price', 'style', 'features']].to_string())
            

    else:
        print("Failed to load data. Exiting.")