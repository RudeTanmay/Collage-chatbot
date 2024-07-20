#this is chat.py
import random
import json
import pickle
import numpy as np

import tensorflow as tf

from model import create_model
from nltk_utils import bag_of_words, tokenize

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pkl"
with open(FILE, 'rb') as f:
    data = pickle.load(f)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model = data["model"]

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = tf.convert_to_tensor(X, dtype=tf.float32)

    output = model(X)

    predicted = tf.argmax(output, axis=1)
    tag = tags[predicted.numpy()[0]]

    probs = tf.nn.softmax(output, axis=1)
    prob = probs[0][predicted[0]]
    if prob.numpy() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "Please check our website for more information..."

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
