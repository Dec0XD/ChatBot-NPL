import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from chatterbot_corpus import corpus_pt_br

import numpy as np
import tensorflow as tf
import random
import json

from flask import Flask, render_template, request
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

with open('intents.json') as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for category in corpus_pt_br.conversations:
    for conversation in category:
        wrds = nltk.word_tokenize(conversation[0])
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(category)
    
        if category not in labels:
            labels.append(category)


for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

model = Sequential()
model.add(Dense(8, input_dim=len(training[0]), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output[0]), activation='softmax'))

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(training, output, epochs=200, batch_size=8, verbose=1)

app = Flask(__name__, template_folder='templates')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    global model, words, labels
    user_text = request.args.get('msg')
    results = model.predict(np.array([np.array([int(w in nltk.word_tokenize(user_text.lower())) for w in words])]))
    results_index = np.argmax(results)
    tag = labels[results_index]

    for tg in data['intents']:
        if tg['tag'] == tag:
            responses = tg['responses']

    return str(random.choice(responses))

if __name__ == '__main__':
    app.run(debug=True)
