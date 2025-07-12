# import json
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import SGD
# import random
# import nltk
# import pickle
# from nltk.stem import LancasterStemmer
# from textblob import TextBlob

# # Initialize stemmer
# stemmer = LancasterStemmer()

# # Load dataset
# with open('chatbot/samudra.json') as file:
#     data = json.load(file)

# # Function to correct typos using TextBlob
# def correct_typo(sentence):
#     blob = TextBlob(sentence)
#     return str(blob.correct())  # Correct spelling errors

# # Prepare data
# words = []
# classes = []
# documents = []
# ignore_words = ['?', '!', '.', ',']

# for intent in data['intents']:
#     for pattern in intent['patterns']:
#         # Correct typos in the sentence
#         corrected_pattern = correct_typo(pattern)
        
#         # Tokenize each word in the sentence
#         word_list = nltk.word_tokenize(corrected_pattern)
#         words.extend(word_list)
#         documents.append((word_list, intent['tag']))
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# # Stem and sort words
# words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
# words = sorted(list(set(words)))

# # Sort classes
# classes = sorted(list(set(classes)))

# # Create training data
# training = []
# output_empty = [0] * len(classes)

# for doc in documents:
#     bag = []
#     pattern_words = doc[0]
#     pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

#     for w in words:
#         bag.append(1) if w in pattern_words else bag.append(0)

#     output_row = list(output_empty)
#     output_row[classes.index(doc[1])] = 1

#     training.append([bag, output_row])

# # Shuffle and convert to np.array
# random.shuffle(training)

# # Ensure consistent shapes before conversion
# bag_length = len(training[0][0])
# output_length = len(training[0][1])

# # Convert to np.array
# train_x = np.array([item[0] for item in training])
# train_y = np.array([item[1] for item in training])

# # Create model
# # Adjusting the model architecture for larger dataset
# model = Sequential()
# model.add(Dense(256, input_shape=(bag_length,), activation='relu'))  # Increased number of units
# model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))  # Increased layer size
# model.add(Dropout(0.5))
# model.add(Dense(output_length, activation='softmax'))

# # Compiling with Adam optimizer instead of SGD
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Lower learning rate for better convergence
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# # Training with a more suitable batch size and epoch count
# model.fit(train_x, train_y, epochs=100, batch_size=32, verbose=1)  # Increased batch size, reduced epochs

# # Save model
# model.save('chatbot/chatbot_samudra1.keras')  # Format HDF5

# # Simpan words dan classes
# with open('chatbot/words-samudraAI1.pkl', 'wb') as f:
#     pickle.dump(words, f)

# with open('chatbot/classes-samudraAI1.pkl', 'wb') as f:
#     pickle.dump(classes, f)

# print("Model trained and saved successfully.")

import json
import pickle
import numpy as np
import nltk
from nltk.stem import LancasterStemmer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random
random.seed(42)
np.random.seed(42)

stemmer = LancasterStemmer()

with open("chatbot/samudra.json") as f:
    data = json.load(f)

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

for intent in data['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

pickle.dump(words, open("chatbot/words-samudraAI_new.pkl", "wb"))
pickle.dump(classes, open("chatbot/classes-samudraAI_new.pkl", "wb"))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(w.lower()) for w in pattern_words]
    bag = [1 if w in pattern_words else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

model.save("chatbot/chatbot_samudra_new.keras")
print("Model trained and saved.")