"""Practical 16 - A property review site wants to predict user sentiment (positive or negative) based on
their written reviews. Build a sentiment classification model using an RNN on simulated
review text data."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Simulated review dataset
reviews = [
    "The house was beautiful and well maintained",
    "Absolutely loved the spacious rooms and clean kitchen",
    "The location was perfect and quiet",
    "Amazing property with great value",
    "Loved the balcony and garden view",
    "Worst house I have seen",
    "Dirty rooms and broken windows",
    "Terrible experience, very noisy neighborhood",
    "Bad condition and overpriced",
    "Poor management and smelly interior"
]

labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1 = Positive, 0 = Negative

# 2. Text preprocessing
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
padded = pad_sequences(sequences, padding='post', maxlen=10)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)

# 4. Build RNN model
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=10),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train model
model.fit(np.array(X_train), np.array(y_train), epochs=10, verbose=1)

# 6. Evaluate
y_pred = (model.predict(np.array(X_test)) > 0.5).astype(int)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
