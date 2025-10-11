# run using streamlit run app.py
# --- ADD THESE IMPORTS AT THE VERY TOP ---
import pandas as pd
import altair as alt
# ----------------------------------------

import streamlit as st

# --- FIX: Move set_page_config to be the FIRST Streamlit command ---
st.set_page_config(page_title="Emotion Detector 🎭", page_icon="🎭", layout="centered")
# -----------------------------------------------------------------

import numpy as np
import tensorflow as tf
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK data quietly
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab',quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ----------------- Load Model + Tokenizer -----------------
MODEL_PATH = "emotion_classifier.keras"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 50


# Use a cache to load the model and tokenizer only once
@st.cache_resource
def load_resources():
    """Loads the pre-trained model and tokenizer."""
    model = load_model(MODEL_PATH, compile=False)
    tokenizer = joblib.load(TOKENIZER_PATH)
    return model, tokenizer


model, tokenizer = load_resources()


def preprocess_text(text):
    """Cleans and preprocesses a single text string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return " ".join(lemmatized_tokens)


# Emotion labels
emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
            'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']

# ----------------- Streamlit UI -----------------

st.title("🎭 Emotion Detection with Deep Learning")
st.markdown("Type a sentence and see the predicted emotions with their probabilities.")

# Input box
user_input = st.text_area("💬 Enter a sentence:", "I am absolutely livid! This is unacceptable!")

if st.button("Analyze"):
    # Check if the input box is empty
    if user_input.strip():  # This is True if the string is not empty
        # Run analysis only if there is text
        processed_input = preprocess_text(user_input)
        seq = tokenizer.texts_to_sequences([processed_input])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

        # Get predictions (logits) and convert to probabilities
        preds = model.predict(padded)
        prediction_probs = tf.sigmoid(preds).numpy()

        predicted_emotions_dict = dict(zip(emotions, prediction_probs[0]))
        sorted_predictions = sorted(predicted_emotions_dict.items(), key=lambda item: item[1], reverse=True)

        # Show results with color-coded bars
        st.subheader("🔎 Prediction Results")
        for emo, prob in sorted_predictions:
            # Color transitions from red (low) to green (high)
            red = int(255 * (1 - prob))
            green = int(255 * prob)
            st.markdown(
                f"""
                <div style="margin-bottom: 5px; padding: 8px; border-radius: 8px; background-color: rgba({red}, {green}, 100, 0.3);">
                    <b>{emo.capitalize()}</b>: {prob * 100:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        # Expandable section for the detailed bar chart
        with st.expander("📊 Show Detailed Chart"):
            results_df = pd.DataFrame(predicted_emotions_dict.items(), columns=['Emotion', 'Probability'])

            chart = alt.Chart(results_df).mark_bar().encode(
                x=alt.X('Probability', axis=alt.Axis(format='%', title='Probability')),
                y=alt.Y('Emotion', sort='-x', title='Emotion'),
                color=alt.Color('Probability', scale=alt.Scale(scheme='redyellowgreen'), legend=None),
                tooltip=['Emotion', alt.Tooltip('Probability', format='.2%')]
            ).properties(
                title='Emotion Analysis Results'
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        # Show a warning if the box is empty
        st.warning("Please enter a sentence to analyze.")
