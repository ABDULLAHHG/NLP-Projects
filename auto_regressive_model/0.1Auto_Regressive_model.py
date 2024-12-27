import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample text data
text = "This is a simple example of an auto-regressive model. It generates text one token at a time."

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Create input sequences and labels
input_sequences = []
for i in range(1, len(text.split())):
    n_gram_sequence = text.split()[:i + 1]
    input_sequences.append(n_gram_sequence)

# Convert sequences to integers
input_sequences = pad_sequences(tokenizer.texts_to_sequences(input_sequences), padding='pre')
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = np.array(y)

# One-hot encode the labels
y = np.eye(total_words)[y]

# Define the model
model = Sequential()
model.add(Embedding(total_words, 50, input_length=X.shape[1]))  # Ensure input length matches X
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, verbose=1)

# Function to generate text
def generate_text(model, tokenizer, seed_text, num_words):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=X.shape[1], padding='pre')  # Match with input length
        
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)
        
        output_word = tokenizer.index_word[predicted_word_index[0]]
        seed_text += " " + output_word
    
    return seed_text

# Generate text
seed_text = "This is"
generated_text = generate_text(model, tokenizer, seed_text, 5)
print(generated_text)

