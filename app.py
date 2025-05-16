# import streamlit as st
# import joblib
# import numpy as np
# import string
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# from textblob import TextBlob
# from scipy.sparse import hstack

# # Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Load models and transformers
# logreg_model = joblib.load("models/logistic_regression_model.pkl")
# rf_model = joblib.load("models/random_forest_model.pkl")
# xgb_model = joblib.load("models/xgboost_model.pkl")
# tfidf = joblib.load("models/tfidf_vectorizer.pkl")
# scaler = joblib.load("models/scaler.pkl")

# # Preprocessing setup
# stop_words = set(stopwords.words("english"))
# lemmatizer = WordNetLemmatizer()

# def preprocess_text(text):
#     text = text.lower().translate(str.maketrans("", "", string.punctuation))
#     tokens = word_tokenize(text)
#     tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and w.isalpha()]
#     return " ".join(tokens)

# def extract_numeric_features(title, text):
#     words = text.split()
#     word_lengths = [len(w) for w in words if w.isalpha()]
#     text_blob = TextBlob(text)
#     title_blob = TextBlob(title)
#     return [
#         np.mean(word_lengths) if word_lengths else 0,                   # avg_word_len
#         text.count("!"),                                                # num_exclam
#         text.count("?"),                                                # num_question
#         title_blob.sentiment.polarity,                                  # sentiment_title_polarity
#         title_blob.sentiment.subjectivity,                              # sentiment_title_subjectivity
#         text_blob.sentiment.polarity,                                   # sentiment_text_polarity
#         text_blob.sentiment.subjectivity,                               # sentiment_text_subjectivity
#         len(set(words)) / len(words) if words else 0,                   # lexical_diversity
#         sum(1 for c in text if c in string.punctuation) / len(text) if text else 0,  # punctuation_ratio
#         sum(1 for w in words if w.lower() in stop_words) / len(words) if words else 0  # stopword_ratio
#     ]

# # Streamlit UI
# st.title("📰 AI-Based Fake News Detection")
# st.markdown("Enter a news **title** and **content**, choose a model, and detect if it's fake or real.")

# model_choice = st.selectbox("Choose a model:", ("Logistic Regression", "Random Forest", "XGBoost"))

# title_input = st.text_input("Enter news title:")
# text_input = st.text_area("Enter news content:")

# if st.button("Predict"):
#     if not title_input.strip() or not text_input.strip():
#         st.warning("Please enter both title and content.")
#     else:
#         combined_text = title_input + " " + text_input
#         clean_text = preprocess_text(combined_text)
#         text_features = tfidf.transform([clean_text])
#         numeric_features = np.array([extract_numeric_features(title_input, text_input)])
#         numeric_features_scaled = scaler.transform(numeric_features)
#         final_features = hstack([text_features, numeric_features_scaled])

#         if model_choice == "Logistic Regression":
#             model = logreg_model
#         elif model_choice == "Random Forest":
#             model = rf_model
#         else:
#             model = xgb_model

#         prediction = model.predict(final_features)[0]
#         label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
#         st.subheader(f"Prediction: {label}")








import streamlit as st
import joblib
import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob
from scipy.sparse import hstack

# Safe NLTK download
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")
    st.stop()

# Load model and transformers with error handling
try:
    xgb_model = joblib.load("models/xgboost_model.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    st.error(f"Failed to load model or vectorizer: {e}")
    st.stop()

# Preprocessing setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    try:
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and w.isalpha()]
        return " ".join(tokens)
    except Exception as e:
        st.error(f"Error in preprocessing text: {e}")
        return ""

def extract_numeric_features(title, text):
    try:
        words = text.split()
        word_lengths = [len(w) for w in words if w.isalpha()]
        text_blob = TextBlob(text)
        title_blob = TextBlob(title)
        return [
            np.mean(word_lengths) if word_lengths else 0,                   # avg_word_len
            text.count("!"),                                                # num_exclam
            text.count("?"),                                                # num_question
            title_blob.sentiment.polarity,                                  # sentiment_title_polarity
            title_blob.sentiment.subjectivity,                              # sentiment_title_subjectivity
            text_blob.sentiment.polarity,                                   # sentiment_text_polarity
            text_blob.sentiment.subjectivity,                               # sentiment_text_subjectivity
            len(set(words)) / len(words) if words else 0,                   # lexical_diversity
            sum(1 for c in text if c in string.punctuation) / len(text) if text else 0,  # punctuation_ratio
            sum(1 for w in words if w.lower() in stop_words) / len(words) if words else 0  # stopword_ratio
        ]
    except Exception as e:
        st.error(f"Error extracting numeric features: {e}")
        return [0]*10  # fallback default features

# Streamlit UI
st.title("📰 AI-Based Fake News Detection")
st.markdown("Enter a news **title** and **content** to detect if it's fake or real using XGBoost.")

title_input = st.text_input("Enter news title:")
text_input = st.text_area("Enter news content:")

if st.button("Predict"):
    if not title_input.strip() or not text_input.strip():
        st.warning("Please enter both title and content.")
    else:
        try:
            combined_text = title_input + " " + text_input
            clean_text = preprocess_text(combined_text)
            if not clean_text:
                st.error("Failed to clean the text. Please check your input.")
                st.stop()

            text_features = tfidf.transform([clean_text])
            numeric_features = np.array([extract_numeric_features(title_input, text_input)])
            numeric_features_scaled = scaler.transform(numeric_features)
            final_features = hstack([text_features, numeric_features_scaled])

            prediction = xgb_model.predict(final_features)[0]
            label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
            st.subheader(f"Prediction: {label}")
        except Exception as e:
            st.error(f"Something went wrong during prediction: {e}")
