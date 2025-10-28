# ReWatch - Watch Recommendation System

ReWatch is a content-based recommendation system that helps users find the perfect watch based on their preferences. The system uses natural language processing (NLP) to parse user input and recommend watches from a curated dataset.

---

## Features

- **Natural Language Input:** Describe your preferences in plain English (e.g., "Suggest me a sports watch from Casio").
- **Content-Based Recommendations:** Matches user preferences with watches based on brand, style, movement, features, and more.
- **Interactive Frontend:** Built with Streamlit for an easy-to-use interface.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/ReWatch.git
   cd ReWatch

   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

   ```

3. Download the spaCy language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Run the Application

1. Start the Streamlit app:

   ```bash
   streamlit run app.py

   ```

2. Open the app in your browser (usually at http://localhost:8501).

3. Enter your preferences in the text box (e.g., "Suggest me a dress watch from Seiko") and click Get Recommendations.

4. View the recommended watches in the table.

### Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.
