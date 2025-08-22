import pandas as pd
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

    # Create a user preference DataFrame from input
    user_df = pd.DataFrame([user_input])

    # Preprocess user input (lowercase and numerical conversion)
    user_df['style'] = user_df['style'].str.lower()
    user_df['movement'] = user_df['movement'].str.lower()
    user_df['strap_material'] = user_df['strap_material'].str.lower()
    user_df['case_diameter'] = float(user_df['case_diameter'])
    user_df['price'] = float(user_df['price'])
    user_df['water_resistance'] = float(user_df['water_resistance'])
    user_df['features'] = user_df['features'].str.lower().str.split(',').apply(lambda x: [f.strip() for f in x])

    # Transform user input using the same trained objects as the original data
    user_encoded = encoder.transform(user_df[['brand', 'style', 'movement', 'strap_material']])
    user_scaled = scaler.transform(user_df[['case_diameter', 'water_resistance', 'price']])
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
