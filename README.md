# Multi-Label Emotion Classification of Tweets

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://emo-bi-lstm-nlp.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)

## Project Overview
This project focuses on the development of a Deep Learning model capable of **Multi-Label Emotion Classification** on tweets. Unlike simple sentiment analysis (positive/negative), this model identifies nuanced emotional states including *anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, and surprise*.

The final model utilizes a **Bidirectional LSTM (Bi-LSTM)** network enhanced with **Multi-Head Attention** and **GloVe embeddings**. A key innovation of this project was the implementation of a **custom weighted loss function** to handle severe class imbalances in the dataset.

 **[Try the Live Demo Here](https://emo-bi-lstm-nlp.streamlit.app/)**

## Dataset
We utilized the **SemEval-2018 Task 1: Affect in Tweets (Subtask E-c)** dataset.
* **Input:** Raw Tweet Text
* **Output:** Multi-label classification across 11 emotion categories.
* **Preprocessing:** The pipeline includes lowercasing, noise removal (regex), tokenization, stop-word removal, and lemmatization.

## Model Architecture
The architecture was optimized through systematic experimentation. The final configuration includes:

1.  **Embedding Layer:** Pre-trained **GloVe (glove.6B.100d)** vectors (Fine-tuned).
2.  **Bi-LSTM Layer:** 128 units, capturing context from both directions.
3.  **Multi-Head Attention:** 4 heads, enabling the model to focus on critical parts of the sentence.
4.  **Global Average Pooling:** Reducing dimensionality.
5.  **Dense Layers:** ReLU activation with Dropout (0.3) for regularization.
6.  **Output Layer:** 11 units (Logits) for the custom weighted loss function.

## Results
I evaluated the model using Macro F1 and Micro F1 scores. The weighted loss function significantly improved the detection of underrepresented emotions (like 'surprise').

| Model Configuration | Macro F1 | Micro F1 |
|:--------------------|:--------:|:--------:|
| Simple RNN | 0.343 | 0.385 |
| Standard LSTM | 0.346 | 0.381 |
| Bi-LSTM (Base) | 0.534 | 0.603 |
| **Bi-LSTM + Attention + Weighted Loss (Final)** | **0.555** | **0.639** |

## Tech Stack
* **Language:** Python
* **Deep Learning:** Keras (Functional API), TensorFlow
* **NLP:** NLTK (Tokenization, Lemmatization)
* **Web Framework:** Streamlit (for the frontend prototype)

## How to Run Locally

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/Johann-75/Emotion-Classifier-using-BiLSTM-RNN.git](https://github.com/Johann-75/Emotion-Classifier-using-BiLSTM-RNN.git)
    cd Emotion-Classifier-using-BiLSTM-RNN
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit App**
    ```bash
    streamlit run app.py
    ```
