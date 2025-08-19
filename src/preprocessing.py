import pandas as pd
import re

def preprocess_data(df):
    """ Performs all necessary data cleaning and preprocessing on the watch DataFrame. """
    if df is None:
        return None
    
    #Conversion to lower case
    for col in ['brand', 'model', 'type', 'movement', 'strap_material', 'style', 'features']:
        if col in df.columns:
            df[col] = df[col].str.lower()

    #Remove non numeric char froms numerical columns
    df['price'] = df['price'].astype(float)

    if 'case_diameter' in df.columns:
        df['case_diameter'] = df['case_diameter'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

    #Handle features column as it has multiple strings
    if 'features' in df.columns:
        df['features'] = df['features'].str.split(',').apply(lambda x: [feature.strip() for feature in x])  

    #Handle missing values
    df.dropna(inplace = True)

    #Add a 'gender' column as it was not in dataset
    if 'gender' not in df.columns:
        df['gender'] = 'unisex' # Default value.

    return df
  

