#train
import numpy as np
import random
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from nltk_utils import bag_of_words, tokenize, lemmatize_word, remove_stopwords

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Process intents
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        w = remove_stopwords(w)
        all_words.extend(w)
        xy.append((w, tag))

# Preprocess words
ignore_words = ['?', '.', '!']
all_words = [lemmatize_word(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique lemmatized words:", all_words)

# Create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

print(input_size, output_size)

# Define the model
model = Sequential([
    Dense(hidden_size, activation='relu', input_shape=(input_size,)),
    Dense(hidden_size, activation='relu'),
    Dense(output_size, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

print(f'Final loss: {history.history["loss"][-1]:.4f}')
print(f'Final accuracy: {history.history["accuracy"][-1]:.4f}')

# Save the model and data
model.save('chatbot_model.h5')

data = {
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

import pickle
with open("training_data.pkl", "wb") as f:
    pickle.dump(data, f)

print('Training complete. Model saved to chatbot_model.h5 and data saved to training_data.pkl')


#chat
import random
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from nltk_utils import bag_of_words, tokenize

# Load intents
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load model and training data
model = load_model('chatbot_model.h5')

import pickle
with open("training_data.pkl", "rb") as f:
    data = pickle.load(f)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = np.array([X])
    
    output = model.predict(X)
    predicted = np.argmax(output, axis=1)

    tag = tags[predicted[0]]

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