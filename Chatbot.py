import streamlit as st
import json
import numpy as np
import random
import nltk
import tensorflow as tf
import time
import os

# Set layout
st.set_page_config(
    page_title="Dashboard Perubahan Iklim",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Membaca file CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Nama user (pastikan kamu sudah definisikan `name` sebelumnya!)
name = "Pengguna"  # Ganti dengan nama dari input user jika perlu
st.write(f'Halo *{name}*, Selamat Datang')

# download modul hanya jika belum tersedia
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=os.path.expanduser("~") + "/nltk_data")

from nltk.stem import LancasterStemmer

# Initialize stemmer
stemmer = LancasterStemmer()

# Load dataset and model
try:
    with open('chatbot.json') as file:
        data = json.load(file)
    model = tf.keras.models.load_model('chatbot_PI_new.keras')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  # Stop execution if the model can't be loaded

# Prepare data
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Stem and sort words
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Functions for processing and predicting
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.5
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        return [{"intent": "default", "probability": "0.0"}]

    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    if not ints or ints[0]['intent'] == "default":
        return "Maaf, saya hanya dapat membantu dengan pertanyaan terkait perubahan iklim. Silakan tanyakan istilah seputar perubahan iklim."

    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        result = "Maaf, saya tidak memiliki jawaban untuk itu. Silakan tuliskan pertanyaan anda seputar istilah perubahan iklim sesuai dengan dokumen Annex IPCC"

    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, data)
    return res

# UI Chat
st.header("Chatbot Istilah Perubahan Iklim")
st.caption("Silahkan ketik pertanyaan anda seputar istilah perubahan iklim.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Fungsi efek streaming respons
def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.1)

# Input user
if prompt := st.chat_input("Silahkan ketik pertanyaan anda di sini.."):
    # Tampilkan pesan user
    with st.chat_message("user"):
        st.markdown(prompt)

    # Simpan ke riwayat chat
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Dapatkan respons dari chatbot
    response = chatbot_response(prompt)

    # Tampilkan respons dengan efek streaming
    with st.chat_message("assistant"):
        response_stream = response_generator(response)
        st.write_stream(response_stream)

    # Simpan respons ke riwayat chat
    st.session_state.messages.append({"role": "assistant", "content": response})
