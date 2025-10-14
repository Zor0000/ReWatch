import pandas as pd
import numpy as np
import re # Import regex for pattern matching
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define comprehensive keyword lists (expand this with more data!)
# NOTE: These need to be based on the values present in your actual 'watch_data.csv'
KNOWN_BRANDS = ['casio', 'seiko', 'rolex', 'omega', 'tag heuer', 'citizen', 'timex']
KNOWN_STYLES = ['sport', 'dress', 'diver', 'pilot', 'field', 'luxury', 'military']
KNOWN_MOVEMENTS = ['automatic', 'quartz', 'mechanical', 'solar']

def create_feature_matrix(df):
    # ... (Keep this function exactly the same) ...
    categorical_features = ['brand', 'style', 'movement', 'strap_material']
    numerical_features = ['case_diameter', 'water_resistance', 'price']
    text_features = 'features'

    # One-Hot Encode categorical data
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # Fit the encoder to the lowercased categorical columns for robustness
    encoded_features = encoder.fit_transform(df[categorical_features].apply(lambda x: x.str.lower()))
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

    # Scale numerical data
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # TF-IDF Vectorize text data (assuming 'features' is a list of strings joined by a space)
    tfidf = TfidfVectorizer()
    # Ensure text features are handled as strings for TFIDF
    tfidf_matrix = tfidf.fit_transform(df[text_features].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

    # Combine all feature vectors
    feature_matrix = pd.concat([df[numerical_features].reset_index(drop=True), encoded_df, tfidf_df], axis=1)

    return feature_matrix, scaler, encoder, tfidf

def inverse_transform_data(df, scaler, numerical_features):
    # ... (Keep this function exactly the same) ...
    transformed_df = df.copy()
    transformed_df[numerical_features] = scaler.inverse_transform(transformed_df[numerical_features])
    return transformed_df


def extract_preferences_from_text(text_query):
    """
    Extracts structured preferences from a natural language query using keywords and regex.
    """
    query = text_query.lower()
    preferences = {
        'brand': '', 'style': '', 'movement': '', 'case_diameter': '',
        'strap_material': '', 'price': '', 'water_resistance': '', 'features': ''
    }

    # 1. Extract Categorical/Text Keywords (Matching to pre-defined lists)
    
    # Brand
    for brand in KNOWN_BRANDS:
        if brand in query:
            preferences['brand'] = brand
            break
            
    # Style
    for style in KNOWN_STYLES:
        if style in query:
            preferences['style'] = style
            break

    # Movement
    for movement in KNOWN_MOVEMENTS:
        if movement in query:
            preferences['movement'] = movement
            break
            
    # 2. Extract Numerical Values (using Regex for digits and units)

    # Case Diameter (looking for a number followed by 'mm')
    # e.g., "38mm" or "around 42 mm"
    case_match = re.search(r'(\d+)\s*mm', query)
    if case_match:
        preferences['case_diameter'] = case_match.group(1)

    # Water Resistance (looking for a number followed by 'meters', 'm', 'bar')
    # e.g., "200m" or "10 bar water resistance"
    water_match = re.search(r'(\d+)\s*(m|meter|bar)', query)
    if water_match:
        preferences['water_resistance'] = water_match.group(1)
        
    # Price (looking for keywords like 'under X', 'cheap', or a specific amount)
    # Simple example: looking for a number near 'dollars', 'usd', or 'under'
    price_match = re.search(r'(?:under|less than)\s*(\$?\d{3,})', query)
    if price_match:
        # Simple extraction - a more complex parser is needed for true price ranges
        preferences['price'] = re.sub(r'[^\d]', '', price_match.group(1)) # Keep only digits

    # 3. Features (Simple extraction: everything that follows 'with', 'and', 'features')
    
    # This is a very simple and fragile approach. A better approach requires advanced NLP.
    # For a simple keyword approach, you could check for specific feature names (e.g., 'chronograph', 'date')
    feature_keywords = ['chronograph', 'date', 'gmt', 'moonphase', 'alarm']
    found_features = [f for f in feature_keywords if f in query]
    if found_features:
        preferences['features'] = ', '.join(found_features)
    
    return preferences


def recommend_watches(user_input, df, feature_matrix, scaler, encoder, tfidf):
    #Generates watch recomm based on user preferences.
    
    # Determine if user_input is a text query or a structured dict
    if isinstance(user_input, str):
        user_preferences = extract_preferences_from_text(user_input)
    elif isinstance(user_input, dict):
        user_preferences = user_input
    else:
        raise ValueError("User input must be a dictionary of preferences or a text string.")

    user_df = pd.DataFrame([user_preferences])

    # --- Start of Feature Vector Creation (remains similar, but applied to parsed dict) ---
    
    # 1. Handle Categorical/Text features for transformation
    for col in ['brand', 'style', 'movement', 'strap_material']:
        val = user_df[col].iloc[0]
        # Only lowercase if not empty
        user_df[col] = val.lower() if val not in ['', None] else ''

    # 2. Handle Numerical features for transformation
    numerical_features = ['case_diameter', 'water_resistance', 'price']
    for col in numerical_features:
        val = user_df[col].iloc[0]
        # Convert to float, defaulting to NaN if empty/None
        user_df[col] = float(val) if val not in ['', None] and str(val).replace('.', '', 1).isdigit() else np.nan

    # 3. Handle Features text input
    features_val = user_df['features'].iloc[0]
    if features_val not in ['', None]:
        # Clean and prepare the features text for TFIDF
        # Assumes features_val is a comma-separated string from the extractor
        user_df['features'] = [' '.join([f.strip().lower() for f in features_val.split(',')])]
    else:
        user_df['features'] = ['']

    # --- Transformation ---
    # The transformation must handle missing (NaN) values for numerical features
    
    # Categorical transformation
    user_encoded = encoder.transform(user_df[['brand', 'style', 'movement', 'strap_material']])
    
    # Numerical transformation (Must be scaled first, then handle NaN/missing data)
    user_scaled_raw = user_df[numerical_features].values
    
    # Scale the non-NaN values
    # NOTE: The scaler will raise an error if it sees NaN, so we must handle it.
    # We will use the same method as in create_feature_matrix: fill NaN with 0 *AFTER* scaling.
    # A cleaner approach is to use a SimpleImputer before the scaler, but let's stick to your current method.
    user_scaled = scaler.transform(user_scaled_raw)
    user_scaled = np.nan_to_num(user_scaled, nan=0.0)  # Replace NaN/missing features with 0.0

    # TFIDF transformation
    user_tfidf = tfidf.transform(user_df['features'].apply(lambda x: ' '.join(x)))

    # Combine user features into a single vector
    user_vector = pd.concat([
        pd.DataFrame(user_scaled, columns=numerical_features),
        pd.DataFrame(user_encoded, columns=encoder.get_feature_names_out(['brand', 'style', 'movement', 'strap_material'])),
        pd.DataFrame(user_tfidf.toarray(), columns=tfidf.get_feature_names_out())
    ], axis=1)

    # --- Calculation ---
    
    # Calculate cosine similarity between user vector and all watch vector
    similarities = cosine_similarity(user_vector, feature_matrix)

    # top 5 most similar watches
    top_indices = similarities.argsort()[0][-5:][::-1]

    # Return recommended watches
    recommended_watches = df.iloc[top_indices]

    # UN-SCALE THE NUMERICAL DATA
    recommended_watches = inverse_transform_data(recommended_watches, scaler, numerical_features)

    return recommended_watches