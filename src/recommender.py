import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_feature_matrix(df):
    #Transforms the preprocessed DataFrame into a numerical feature matrix.
    categorical_features = ['brand', 'style', 'movement', 'strap_material']
    numerical_features = ['case_diameter', 'water_resistance', 'price']
    text_features = 'features'

    # One-Hot Encode categorical data
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

    # Scale numerical data
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # TF-IDF Vectorize text data
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df[text_features].apply(lambda x: ' '.join(x)))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

    # Combine all feature vectors
    feature_matrix = pd.concat([df[numerical_features].reset_index(drop=True), encoded_df, tfidf_df], axis=1)

    return feature_matrix, scaler, encoder, tfidf

def inverse_transform_data(df, scaler, numerical_features):
    #Inverse transforms specified numerical columns of a DataFrame using a fitted scaler.
    # Create a copy to avoid modifying the original DataFrame
    transformed_df = df.copy()
    
    # Inverse transform the specified numerical features
    transformed_df[numerical_features] = scaler.inverse_transform(transformed_df[numerical_features])
    
    return transformed_df

def recommend_watches(user_input, df, feature_matrix, scaler, encoder, tfidf):
    #Generates watch recomm based on user preferences.
    """
    Args:
        user_input (dict): A dictionary containing user preferences.
        df (pd.DataFrame): The preprocessed watch data.
        feature_matrix (pd.DataFrame): The precomputed feature matrix of all watches.
        scaler: The trained StandardScaler for numerical data.
        encoder: The trained OneHotEncoder for categorical data.
        tfidf: The trained TfidfVectorizer for text data.

    Returns:
        pd.DataFrame: A DataFrame of the top N recommended watches.
    """
    user_df = pd.DataFrame([user_input])

    # Handle empty categorical/text input
    for col in ['brand', 'style', 'movement', 'strap_material']:
        val = user_df[col].iloc[0]
        user_df[col] = val.lower() if val not in ['', None] else ''

    # Handle empty numerical input
    for col in ['case_diameter', 'price', 'water_resistance']:
        val = user_df[col].iloc[0]
        user_df[col] = float(val) if val not in ['', None] else np.nan

    # Handle empty features input
    features_val = user_df['features'].iloc[0]
    if features_val not in ['', None]:
        # Only apply .lower() and .split(',') if it's a string
        user_df['features'] = [f.strip().lower() for f in features_val.split(',')]
    else:
        user_df['features'] = ['']

    # Transform user input using the same trained objects as the original data
    user_encoded = encoder.transform(user_df[['brand', 'style', 'movement', 'strap_material']])
    user_scaled = scaler.transform(user_df[['case_diameter', 'water_resistance', 'price']])
    user_scaled = np.nan_to_num(user_scaled, nan=0.0)  # Replace NaN with 0.0
    user_tfidf = tfidf.transform(user_df['features'].apply(lambda x: ' '.join(x)))

    # Combine user features into a single vector
    user_vector = pd.concat([
        pd.DataFrame(user_scaled, columns=['case_diameter', 'water_resistance', 'price']),
        pd.DataFrame(user_encoded, columns=encoder.get_feature_names_out(['brand', 'style', 'movement', 'strap_material'])),
        pd.DataFrame(user_tfidf.toarray(), columns=tfidf.get_feature_names_out())
    ], axis=1)

    #Calculate cosine similarity between user vector and all watch vector
    similarities = cosine_similarity(user_vector, feature_matrix)

    #top 5 most similar watches
    top_indices = similarities.argsort()[0][-5:][::-1]

    #REturn recommended watches
    recommended_watches = df.iloc[top_indices]

    # UN-SCALE THE NUMERICAL DATA
    numerical_features = ['case_diameter', 'water_resistance', 'price']
    recommended_watches = inverse_transform_data(recommended_watches, scaler, numerical_features)

    return recommended_watches
