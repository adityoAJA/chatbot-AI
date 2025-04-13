import streamlit as st
import random
import time
import re

# Intent data
intents = [
    {
        "tag": "sapaan",
        "patterns": ["hai", "halo", "apa kabar", "selamat pagi", "selamat sore"],
        "responses": [
            "Halo! Bagaimana saya bisa membantu Anda hari ini?",
            "Hai! Ada yang bisa saya bantu mengenai perubahan iklim?",
            "Selamat datang! Bagaimana saya bisa membantu?"
        ],
    },
    {
        "tag": "perubahan_iklim",
        "patterns": [
            "apa itu perubahan iklim",
            "apa penyebab perubahan iklim",
            "apa dampak perubahan iklim",
            "cara mitigasi perubahan iklim",
            "bagaimana proyeksi iklim di indonesia"
        ],
        "responses": [
            "Perubahan iklim adalah perubahan jangka panjang dalam pola cuaca rata-rata di suatu wilayah atau di seluruh dunia. Perubahan ini disebabkan oleh aktivitas manusia, seperti pembakaran bahan bakar fosil, deforestasi, dan aktivitas industri yang meningkatkan konsentrasi gas rumah kaca di atmosfer.",
            "Perubahan iklim terutama disebabkan oleh peningkatan emisi gas rumah kaca, seperti karbon dioksida (CO2) dan metana (CH4). Gas-gas ini dihasilkan dari aktivitas manusia, termasuk pembakaran bahan bakar fosil, deforestasi, dan pertanian intensif.",
            "Perubahan iklim berdampak pada peningkatan suhu global, kenaikan permukaan laut, perubahan pola curah hujan, dan peningkatan frekuensi cuaca ekstrem. Dampak ini juga mempengaruhi ekosistem, keanekaragaman hayati, dan kesehatan manusia.",
            "Untuk mengurangi dampak perubahan iklim, kita dapat melakukan berbagai langkah mitigasi, seperti beralih ke energi terbarukan, meningkatkan efisiensi energi, mengurangi deforestasi, dan mendukung kebijakan lingkungan yang berkelanjutan.",
            "Proyeksi iklim di Indonesia menunjukkan bahwa suhu akan semakin menghangat, dengan potensi peningkatan kejadian cuaca ekstrem."
        ],
    },
    # Tambahkan intents lainnya sesuai kebutuhan...
]

st.header("Chatbot Perubahan Iklim")
st.caption("Silahkan ketik pertanyaan anda seputar perubahan iklim di bawah ini.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to clean and normalize text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Function to match user input to an intent
def get_response(user_input):
    cleaned_input = clean_text(user_input)
    for intent in intents:
        for pattern in intent["patterns"]:
            if pattern in cleaned_input:
                return random.choice(intent["responses"])
    return "Maaf, saya tidak mengerti. Bisa tolong jelaskan lebih lanjut?"

# Fungsi untuk mensimulasikan efek streaming respons
def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.1)  # Mengatur kecepatan efek streaming

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response based on intent matching
    response = get_response(prompt)

    # Display assistant response in chat message container with streaming effect
    with st.chat_message("assistant"):
        response_stream = response_generator(response)
        st.write_stream(response_stream)

    # Add final assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
