import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
# import nltk_downloader
from nltk.stem import LancasterStemmer
from tensorflow.keras.models import load_model
from functions import sla_plotter
from functions import narrative  # pada folder functions yg sama dengan sla_plotter
import re
import time

# Streamlit UI
# st.set_page_config(page_title="SAMUDRA-AI Chatbot", page_icon="🌊")

st.title("🤖 Chatbot SAMUDRA-AI 🌊")
st.markdown("Tanyakan apa saja tentang tinggi muka laut (TML), proyeksi, atau kondisi per desa!")

stemmer = LancasterStemmer()

# Load resources
with open("samudra.json") as f:
    intents = json.load(f)

words = pickle.load(open("words-samudraAI_new.pkl", "rb"))
classes = pickle.load(open("classes-samudraAI_new.pkl", "rb"))
model = load_model("chatbot_samudra_new.keras")

# Utility functions
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

def stream_response(text, delay=0.03):
    for word in text.split():
        yield word + " "
        time.sleep(delay)

# Reset chat history button
if st.button("🪑 Reset Chat"):
    st.session_state.messages = []
    st.rerun()

# Inisialisasi session_state untuk menyimpan pesan
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan seluruh riwayat percakapan
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["type"] == "text":
            st.markdown(msg["content"])
        elif msg["type"] == "chart":
            st.plotly_chart(msg["content"])

# Input pengguna
user_input = st.chat_input("Tulis pertanyaanmu di sini...")

if user_input:
    st.session_state.messages.append({"role": "user", "type": "text", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        input_bow = bow(user_input, words)
        res = model.predict(np.array([input_bow]))[0]
        idx = np.argmax(res)
        tag = classes[idx]

        if res[idx] > 0.6:
            for intent in intents['intents']:
                if intent['tag'] == tag:
                    response_text = random.choice(intent['responses'])

                    st.write_stream(stream_response(response_text))
                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": response_text})

                    function_name = intent.get("function")
                    if function_name:
                        fig = None
                        narration = None

                        if function_name == "plot_tml_desa":
                            desa_match = re.search(r"desa\s+([\w\s]+)", user_input.lower())
                            tahun_match = re.search(r"tahun\s+(\d{4})", user_input.lower())
                            if desa_match:
                                desa_name = desa_match.group(1).strip()
                                tahun = int(tahun_match.group(1)) if tahun_match else None
                                fig, df = sla_plotter.plot_tml_desa(desa_name, tahun=tahun, return_df=True)
                                if fig:
                                    narration = narrative.generate_narrative("plot_tml_desa", desa=desa_name, df=df)

                        elif function_name == "plot_tml_tahunan":
                            tahun_match = re.search(r"tahun\s+(\d{4})", user_input.lower())
                            if tahun_match:
                                tahun = tahun_match.group(1)
                                fig, df = sla_plotter.plot_tml_tahunan(tahun, return_df=True)
                                if fig:
                                    narration = narrative.generate_narrative("plot_tml_tahunan", tahun=tahun, df=df)

                        elif function_name == "tren_tml_desa":
                            desa_match = re.search(r"desa\s+([\w\s]+)", user_input.lower())
                            if desa_match:
                                desa_name = desa_match.group(1).strip()
                                fig, df, trend = sla_plotter.tren_tml_desa(desa_name, return_df=True)
                                if fig:
                                    narration = narrative.generate_narrative("tren_tml_desa", desa=desa_name, trend=trend)

                        elif function_name == "tren_tml_nasional":
                            fig, df, trend = sla_plotter.tren_tml_nasional(return_df=True)
                            if fig:
                                narration = narrative.generate_narrative("tren_tml_nasional", trend=trend)

                        elif function_name == "plot_bandingkan_desa":
                            desa_matches = re.findall(r"desa\s+([\w\s]+)", user_input.lower())
                            if len(desa_matches) >= 2:
                                desa1, desa2 = desa_matches[:2]
                                fig, df = sla_plotter.plot_bandingkan_desa(desa1, desa2, return_df=True)
                                if fig:
                                    narration = narrative.generate_narrative("plot_bandingkan_desa", desa1=desa1, desa2=desa2, df=df)

                        elif function_name == "ranking_tml_desa":
                            fig = sla_plotter.ranking_tml_desa(user_input)
                            if fig is not None:
                                narration = narrative.generate_narrative("ranking_tml_desa", df=fig)
                                st.dataframe(fig, hide_index=True)
                                fig = None  # do not plot chart

                        elif function_name == "ranking_tml_provinsi":
                            fig = sla_plotter.ranking_tml_provinsi()
                            if fig is not None:
                                narration = narrative.generate_narrative("ranking_tml_provinsi", df=fig)
                                st.dataframe(fig, hide_index=True)
                                fig = None

                        elif function_name == "peta_tml_tahun":
                            tahun_match = re.search(r"tahun\s+(\d{4})", user_input.lower())
                            if tahun_match:
                                tahun = int(tahun_match.group(1))
                                with st.spinner("📸 Menyiapkan peta TML tahun..."):
                                    fig, region_max, region_min, prov_max, prov_min = sla_plotter.peta_tml_tahun(tahun, return_regions=True)
                                    narration = narrative.generate_narrative("peta_tml_tahun", tahun=tahun,
                                                                             region_max=region_max, region_min=region_min,
                                                                             prov_max=prov_max, prov_min=prov_min)

                        elif function_name == "peta_tren_tml_nasional":
                            with st.spinner("📊 Menyiapkan peta tren nasional..."):
                                fig, region_max, region_min, prov_max, prov_min = sla_plotter.peta_tren_tml_nasional(return_regions=True)
                                narration = narrative.generate_narrative("peta_tren_tml_nasional",
                                                                         region_max=region_max, region_min=region_min,
                                                                         prov_max=prov_max, prov_min=prov_min)

                        elif function_name == "tren_tml_kabupaten":
                            match = re.search(r"kabupaten\s+([\w\s]+)", user_input.lower())
                            if match:
                                kab = match.group(1).strip()
                                fig, df, trend = sla_plotter.tren_tml_kabupaten(kab, return_df=True)
                                if fig:
                                    narration = narrative.generate_narrative("tren_tml_kabupaten", kabupaten=kab, trend=trend)

                        elif function_name == "tren_tml_kecamatan":
                            match = re.search(r"kecamatan\s+([\w\s]+)", user_input.lower())
                            if match:
                                kec = match.group(1).strip()
                                fig, df, trend= sla_plotter.tren_tml_kecamatan(kec, return_df=True)
                                if fig:
                                    narration = narrative.generate_narrative("tren_tml_kecamatan", kecamatan=kec, trend=trend)

                        elif function_name == "grafik_tahunan_kabupaten":
                            match_kab = re.search(r"kabupaten\s+([\w\s]+)", user_input.lower())
                            match_thn = re.search(r"tahun\s+(\d{4})", user_input.lower())
                            if match_kab and match_thn:
                                kab = match_kab.group(1).strip()
                                tahun = match_thn.group(1)
                                fig, df = sla_plotter.grafik_tahunan_kabupaten(kab, tahun, return_df=True)
                                if fig:
                                    narration = narrative.generate_narrative("grafik_tahunan_kabupaten", kabupaten=kab, tahun=tahun, df=df)

                        elif function_name == "grafik_tahunan_kecamatan":
                            match_kec = re.search(r"kecamatan\s+([\w\s]+)", user_input.lower())
                            match_thn = re.search(r"tahun\s+(\d{4})", user_input.lower())
                            if match_kec and match_thn:
                                kec = match_kec.group(1).strip()
                                tahun = match_thn.group(1)
                                fig, df = sla_plotter.grafik_tahunan_kecamatan(kec, tahun, return_df=True)
                                if fig: 
                                    narration = narrative.generate_narrative("grafik_tahunan_kecamatan", kecamatan=kec, tahun=tahun, df=df)

                        if fig:
                            st.plotly_chart(fig)
                            st.session_state.messages.append({"role": "assistant", "type": "chart", "content": fig})

                        if narration:
                            st.markdown(narration)
                            st.session_state.messages.append({"role": "assistant", "type": "text", "content": narration})
                    break
        else:
            warning_text = "Maaf, saya tidak memahami pertanyaan Anda."
            st.warning(warning_text)
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": warning_text})