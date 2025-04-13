import json
import pandas as pd
import streamlit as st

# Membaca file JSON
with open('chatbot\chatbot.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Menginisialisasi list kosong untuk menyimpan tag dan responses
tags = []
responses = []

# Menelusuri data JSON dan menambahkan ke list
for intent in data['intents']:
    tag = intent['tag']
    response_list = intent['responses']
    for response in response_list:
        tags.append(tag)
        responses.append(response)

# Membuat DataFrame dari list tags dan responses
df = pd.DataFrame({'Istilah': tags, 'Penjelasan': responses})

# Menghapus index dari DataFrame
df_reset = df.reset_index(drop=True)

# Menambahkan kolom index baru yang dimulai dari 1
df_reset.index = df_reset.index + 1

# Menampilkan tabel di Streamlit
st.header("Tabel Istilah Perubahan Iklim")
# Menampilkan tabel tanpa index menggunakan st.table()
st.table(df_reset)
