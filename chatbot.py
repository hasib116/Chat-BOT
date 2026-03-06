# AI Chatbot using TensorFlow
# simple intent based chatbot

import json
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -----------------------------
# Training Data
# -----------------------------

intents = {
    "intents": [

        {
            "tag": "greeting",
            "patterns": [
                "hi",
                "hello",
                "hey",
                "good morning",
                "good evening",
                "how are you"
            ],
            "responses": [
                "Hello!",
                "Hi there!",
                "Hey!",
                "Nice to meet you!"
            ]
        },

        {
            "tag": "name",
            "patterns": [
                "what is your name",
                "who are you",
                "tell me your name"
            ],
            "responses": [
                "I am an AI chatbot.",
                "You can call me TensorBot.",
                "My name is TensorFlow Bot."
            ]
        },

        {
            "tag": "help",
            "patterns": [
                "can you help me",
                "i need help",
                "help",
                "support"
            ],
            "responses": [
                "Sure I can help.",
                "Tell me your problem.",
                "What do you need help with?"
            ]
        },

        {
            "tag": "python",
            "patterns": [
                "what is python",
                "tell me about python",
                "python programming"
            ],
            "responses": [
                "Python is a popular programming language.",
                "Python is widely used for AI and automation.",
                "Python is simple and powerful."
            ]
        },

        {
            "tag": "bye",
            "patterns": [
                "bye",
                "goodbye",
                "see you later"
            ],
            "responses": [
                "Goodbye!",
                "See you later!",
                "Have a nice day!"
            ]
        }

    ]
}


# -----------------------------
# Prepare Data
# -----------------------------

sentences = []
labels = []
responses = {}
classes = []

for intent in intents["intents"]:

    tag = intent["tag"]
    classes.append(tag)
    responses[tag] = intent["responses"]

    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(tag)


# -----------------------------
# Text Tokenization
# -----------------------------

tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

max_len = max(len(x) for x in sequences)

padded = pad_sequences(sequences, maxlen=max_len, padding="post")


# -----------------------------
# Label Encoding
# -----------------------------

label_index = {}
index_label = {}

for i, tag in enumerate(classes):
    label_index[tag] = i
    index_label[i] = tag

y = np.array([label_index[label] for label in labels])


# -----------------------------
# Build Neural Network
# -----------------------------

model = Sequential()

model.add(Dense(128, input_shape=(max_len,), activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(len(classes), activation="softmax"))


# -----------------------------
# Compile Model
# -----------------------------

sgd = SGD(learning_rate=0.01, momentum=0.9)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=sgd,
    metrics=["accuracy"]
)


# -----------------------------
# Train Model
# -----------------------------

print("Training chatbot...")

model.fit(
    padded,
    y,
    epochs=300,
    batch_size=8,
    verbose=0
)

print("Training completed!")


# -----------------------------
# Chatbot Response Function
# -----------------------------

def get_response(user_input):

    seq = tokenizer.texts_to_sequences([user_input])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding="post")

    prediction = model.predict(padded_seq, verbose=0)

    index = np.argmax(prediction)

    tag = index_label[index]

    response_list = responses[tag]

    return random.choice(response_list)


# -----------------------------
# Chat Loop
# -----------------------------

print("\nAI Chatbot is ready!")
print("Type 'quit' to exit.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Bot: Goodbye!")
        break

    answer = get_response(user_input)

    print("Bot:", answer)


# -----------------------------
# Save Model
# -----------------------------

model.save("chatbot_model.h5")

print("Model saved successfully!")