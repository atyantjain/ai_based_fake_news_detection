# AI based fake news detection
This project is towards the completion of Group Assignment 3 of the subject '36121 Artificial Intelligence Principles and Applications - Autumn 2025'

This project is an end-to-end implementation of an AI-powered system to detect fake news articles using Natural Language Processing (NLP) and Machine Learning techniques. It includes model training, evaluation, deployment code, and visualization for insights.

## Project Structure
```text
ai_based_fake_news_detection/
│
├── app.py                           # Streamlit app
├── AI_2 (1).ipynb                   # Jupyter notebook with data processing & model training
├── models/
│   ├── scaler.pkl                   # Standard scaler for numeric features
│   ├── tfidf_vectorizer.pkl         # TF-IDF vectorizer for text preprocessing
│   └── xgboost_model.pkl            # Trained XGBoost classifier
├── Graphs/
│   ├── *.svg, *.png                 # Visualizations: feature importance, distributions, etc.
├── .devcontainer/
│   └── devcontainer.json            # VS Code remote container setup
└── README.md                        # Project overview and instructions
```

## Problem Statement
With the rise of misinformation on the internet, it is increasingly important to develop tools that can automatically distinguish between fake and real news articles. This project aims to build an AI model that can classify news content as Fake or Real based on its textual content.

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- NLTK, TextBlob
- TF-IDF Vectorization
- Streamlit/Flask (for web interface)
- Matplotlib, Seaborn (for visualizations)
- VS Code DevContainer (for environment setup)

## Features
- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- NLTK, TextBlob
- TF-IDF Vectorization
- Streamlit/Flask (for web interface)
- Matplotlib, Seaborn (for visualizations)
- VS Code DevContainer (for environment setup)

## How to run the application
1. Clone the repository 
- git clone https://github.com/your-username/ai_based_fake_news_detection.git
- cd ai_based_fake_news_detection

2. Install the dependencies
- pip install -r requirements.txt

3. Run the application
- streamlit run app.py

4. View the web interface at http://localhost:8501/

## Models
The models/ folder includes:
- scaler.pkl – for feature scaling
- tfidf_vectorizer.pkl – for vectorizing input text
- xgboost_model.pkl – trained classification model

## Visual Insights
Graphs in the Graphs/ folder illustrate:
- Feature importance from XGBoost
- Distribution of text polarity and subjectivity
- Presence of questions, punctuation, and more

## Authors
- Atyant Jain (25156985)
- Kittituch Wongwatcharapaiboon (25544646)
- Peeranont Dongpakkij (24666594)
- Ratticha Ratanawarocha (24996427)
- Vaibhav Kiran Ghaisas (25415629)