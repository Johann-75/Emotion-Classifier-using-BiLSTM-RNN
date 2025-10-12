import pandas as pd
import numpy as np
import re
import nltk
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    Attention,
    MultiHeadAttention,
    GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Preprocessing
def preprocess_text(text):
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

def main():
    #  Load Data 
    df = pd.read_csv("2018-E-c-En-train.txt", sep="\t")
    df['processed_text'] = df['Tweet'].apply(preprocess_text)

    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
                'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']

    X = df['processed_text'].values
    y = df[emotions].values  # multi-label

    #  Tokenizer 
    voc_size = 5000 # Limits the vocabulary to the top 5,000 most frequent words.
    tokenizer = Tokenizer(num_words=voc_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X) # Converts each tweet into a list of integers (word indices) assigned by the tokenizer
    joblib.dump(tokenizer, "tokenizer.pkl")

    #  Load pretrained GloVe Embeddings 
    glove_path = "glove.6B.100d.txt"  
    embedding_index = {}
    embedding_dim = 100

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vector

    embedding_matrix = np.zeros((voc_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i >= voc_size:
            continue
        vector = embedding_index.get(word)
        if vector is not None:
            embedding_matrix[i] = vector

    # Padding Sequences
    sent_length = 50
    X_padded = pad_sequences(sequences, padding='post', maxlen=sent_length) # put zeros in extra space

    #  Train/Test Split 
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, test_size=0.2, random_state=42
    )

    #Now we want to assign certain weights to each emotion based on their frequency in the dataset, to avoid bias towards common/always occurring emotions
    
    pos_weights = []
    for i in range(y_train.shape[1]):  # For each of the 11 emotions
        # Count negatives (0) and positives (1)
        neg = np.sum(y_train[:, i] == 0)
        pos = np.sum(y_train[:, i] == 1)
        # The weight is the ratio of negatives to positives
        pos_weights.append(neg / (pos + 1e-6))  # Added a small epsilon to avoid division by zero

    # Weighted loss function passed to model.compile
    def weighted_binary_crossentropy(y_true, y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(
            labels=tf.cast(y_true, tf.float32),
            logits=y_pred,
            pos_weight=tf.constant(pos_weights, dtype=tf.float32)
        )

    #  FINAL MODEL
    
    inputs = Input(shape=(sent_length,))
    x = Embedding(input_dim=voc_size, output_dim=embedding_dim,
                  weights=[embedding_matrix], input_length=sent_length, trainable=True)(inputs)
    x = Dropout(0.3)(x)
    # x = SimpleRNN(128)(x)
    # x = LSTM(128, return_sequences=True)(x)
    # x = LSTM(64, return_sequences=False)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(
        x)  # Produces a hidden state at each time step that encodes information from the past (forward) and future (backward).
    # x = Attention()([x,
    #                  x])  # attention lets the model look at all hidden states of the BiLSTM. It learns weights over time, which words matter most for the current task, Then it produces a weighted sum of the hidden states: words that matter more get higher weight, less important words get lower weight.
    x = MultiHeadAttention(num_heads=4, key_dim=128)(x, x) #multiple perspectives, which words matter in different ways simultaneously
    x = GlobalAveragePooling1D()(
        x)  # averages the features across all time steps obtained from the 3d tensor (batch_size, timestep, features) producing a 2d tensor (batch_size, features) for the next layer
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(11)(x)  #we want to get the logits values for calculating loss more effectively, later we will calculate probabilities using sigmoid
    # outputs = Dense(11, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    optimizer = Adam(learning_rate=1e-3)
    model.compile(loss=weighted_binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    #  Training 
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=30,
                        callbacks=EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)  # check the last 3 epochs to check for any actual improvement, else quit
    )

    #  Threshold Optimization 
    y_pred_probs = model.predict(X_test)
    best_thresholds = []
    for i, emotion in enumerate(emotions):
        precision, recall, thresholds = precision_recall_curve(y_test[:, i], y_pred_probs[:, i])
        f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
        best_thresh = thresholds[np.argmax(f1_scores[:-1])]
        best_thresholds.append(best_thresh)
        print(emotion, "best threshold:", best_thresh)

    # Apply thresholds per emotion
    y_pred = np.zeros_like(y_pred_probs)
    for i, thresh in enumerate(best_thresholds):
        y_pred[:, i] = (y_pred_probs[:, i] > thresh).astype(int)

    #  F1 Scores 
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    print("Macro F1:", f1_macro)
    print("Micro F1:", f1_micro)

    model.save("emotion_classifier.keras")

    #  Test Prediction 
    test_sentence = "I am absolutely livid! This is unacceptable!"
    # test_sentence = "Wait, what? I didn’t see that coming at all!"
    processed_test_sentence = preprocess_text(test_sentence)
    test_sequence = tokenizer.texts_to_sequences([processed_test_sentence])
    test_padded = pad_sequences(test_sequence, padding='post', maxlen=sent_length)

    prediction = model.predict(test_padded)
    prediction_probs = tf.sigmoid(prediction).numpy()
    predicted_emotions_dict = dict(zip(emotions, prediction_probs[0]))
    sorted_predictions = sorted(predicted_emotions_dict.items(), key=lambda item: item[1], reverse=True)
    print(f"\nPredicted emotions for: {test_sentence}\n")
    for emotion, prob in sorted_predictions:
        print(f"{emotion}: {prob*100:.2f}%")

if __name__ == "__main__":
    main()

