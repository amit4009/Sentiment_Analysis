import pandas as pd
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Activation, Dropout, Dense, Flatten, GlobalMaxPooling1D, Embedding, Conv1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, render_template
import nltk
from nltk.corpus import stopwords
import io
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load stopwords
stopwords_list = set(stopwords.words('english'))
maxlen = 100

# Load LSTM model
model_path = 'lstm_model.h5'
pretrained_lstm_model = load_model(model_path)

# Load tokenizer from file using pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Import custom preprocessing function
from b2_preprocessing_function import CustomPreprocess
custom = CustomPreprocess()

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    query_asis = [str(x) for x in request.form.values()]

    # Preprocess review text with earlier defined preprocess_text function
    query_processed_list = []
    for query in query_asis:
        query_processed = custom.preprocess_text(query)
        query_processed_list.append(query_processed)

    # Tokenizing instance with earlier trained tokenizer
    query_tokenized = tokenizer.texts_to_sequences(query_processed_list)

    # Padding instance to have a maximum length of 100 tokens
    query_padded = pad_sequences(query_tokenized, padding='post', maxlen=maxlen)

    # Pass tokenized instance to the LSTM model for predictions
    query_sentiments = pretrained_lstm_model.predict(query_padded)

    # Determine sentiment based on prediction
    if query_sentiments[0][0] > 0.5:
        prediction_text = f"Positive Review with probable IMDb rating as: {np.round(query_sentiments[0][0]*10, 1)}"
    else:
        prediction_text = f"Negative Review with probable IMDb rating as: {np.round(query_sentiments[0][0]*10, 1)}"

    # Prepare JSON response
    response = {'prediction_text': prediction_text}

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, threaded=False)  # Ensure threading is disabled if causing issues
