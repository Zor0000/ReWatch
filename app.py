import streamlit as st
import pandas as pd
from src.recommender import recommend_watches, create_feature_matrix, extract_preferences_from_text

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/watch_data.csv")
    preprocessed_df = df.copy()  # Add your preprocessing logic here
    feature_matrix, scaler, encoder, tfidf = create_feature_matrix(preprocessed_df)
    return preprocessed_df, feature_matrix, scaler, encoder, tfidf

# Load data
preprocessed_df, feature_matrix, scaler, encoder, tfidf = load_data()

# Streamlit app layout
st.title("ReWatch")
st.write("Describe your watch preferences, and we'll recommend the best watches for you!")

# User input
user_input = st.text_input("Enter your preferences (e.g., 'Suggest me a sports watch from Casio'):")

if st.button("Get Recommendations"):
    if user_input.strip():
        # Parse user input into preferences using extract_preferences_from_text
        user_preferences = extract_preferences_from_text(user_input)

        # Get recommendations
        recommendations = recommend_watches(user_preferences, preprocessed_df, feature_matrix, scaler, encoder, tfidf)

        # Display recommendations
        st.write("### Recommended Watches:")
        st.dataframe(recommendations[['brand', 'model', 'price', 'style', 'features']])
    else:
        st.warning("Please enter your preferences to get recommendations.")