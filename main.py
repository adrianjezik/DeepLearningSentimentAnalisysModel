import pyexcel_ods3
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import logging

logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

file_path_train = 'DaneTreningowe.ods'
data_train = pyexcel_ods3.get_data(file_path_train)
first_sheet_train_name = list(data_train.keys())[0]
df_train = pd.DataFrame(data_train[first_sheet_train_name])

label_encoder = LabelEncoder()
df_train.iloc[:, 1] = label_encoder.fit_transform(df_train.iloc[:, 1])

max_words = 2000
max_length = 100
tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
tokenizer.fit_on_texts(df_train.iloc[:, 0].astype(str))
X_train_seq = tokenizer.texts_to_sequences(df_train.iloc[:, 0].astype(str))
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)

model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=256, input_length=max_length))
model.add(LSTM(units=100, dropout=0.2))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train_pad, X_val_pad, y_train, y_val = train_test_split(X_train_pad, df_train.iloc[:, 1].astype('float32'), test_size=0.2, random_state=42)

# ModelCheckpoint
checkpoint_callback = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# TensorBoard zapisuje logi do katalogu 'logs'
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

model.fit(X_train_pad, y_train, epochs=10, batch_size=64, validation_data=(X_val_pad, y_val), callbacks=[checkpoint_callback, tensorboard_callback])

logging.info('Model trained successfully.')

file_path_test = 'DaneTestowe.ods'
data_test = pyexcel_ods3.get_data(file_path_test)
first_sheet_test_name = list(data_test.keys())[0]
df_test = pd.DataFrame(data_test[first_sheet_test_name])

df_test.iloc[:, 1] = label_encoder.transform(df_test.iloc[:, 1])
X_test_seq = tokenizer.texts_to_sequences(df_test.iloc[:, 0].astype(str))
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)

loss, accuracy = model.evaluate(X_test_pad, df_test.iloc[:, 1].astype('float32'))
print(f'Test Accuracy: {accuracy * 100:.2f}%')


logging.info(f'Test Accuracy: {accuracy * 100:.2f}%')

# Przykładowe opinie
sample_reviews = [
    "This book was extraordinary, it fascinated me from the first to the last page.",
    "Well written, surprising plot twist, highly recommended!",
    "Unfortunately, this book disappointed me. I had higher expectations.",
    "Poorly written, the plot didn't keep me in suspense.",
    "A masterpiece of literature! I couldn't put it down, I really recommend it."
]

sample_seqs = tokenizer.texts_to_sequences(sample_reviews)
sample_pads = pad_sequences(sample_seqs, maxlen=max_length)
predictions = model.predict(np.array(sample_pads))

predicted_sentiments = [1 if prediction > 0.90 else 0 for prediction in predictions]

# Wyświetlenie wyników
for i, review in enumerate(sample_reviews):
    sentiment_label = "Pozytywna" if predicted_sentiments[i] > 0.9 else "Negatywna"
    print(f"Opinia {i + 1}: {review}")
    print(f"Opinia zaproponowana przez wytrenowany model: {sentiment_label}\n")
    # Logowanie wyników dla każdej opinii
    logging.info(f"Opinia {i + 1}: {review}")
    logging.info(f"Opinia zaproponowana przez wytrenowany model: {sentiment_label}")
