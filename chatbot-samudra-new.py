# chatbot-samudra-new.py

import streamlit as st
import json
import re
import time
import pandas as pd
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import all your function modules
from functions import plotter, narrative, narrative_proj

# =============================================================================
# 1. ASSET LOADING (Cached for performance)
# =============================================================================

@st.cache_resource
def load_resources(model_path="bert-samudra-model"):
    """
    Loads all required assets: BERT model, tokenizer, and intents data.
    This function runs only once thanks to @st.cache_resource.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    with open("samudra.json", encoding="utf-8") as f:
        intents = json.load(f)
    
    with open("label_map.json", "r") as f:
        label_map = json.load(f)
        # We need a mapping from ID to Label (tag) for prediction output
        id2label = {v: k for k, v in label_map.items()}
        
    return tokenizer, model, intents, id2label

# =============================================================================
# 2. CORE LOGIC FUNCTIONS
# =============================================================================

def predict_intent(text, tokenizer, model, id2label):
    """
    Predicts the intent (tag) from input text using the BERT model.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_class_id = logits.argmax().item()
    confidence = torch.softmax(logits, dim=1)[0][predicted_class_id].item()
    
    tag = id2label.get(predicted_class_id, "fallback") # Use .get for safety
    
    return tag, confidence

def extract_entities(text):
    """
    VERSI FINAL: Mengekstrak entitas dengan memprioritaskan pola perbandingan 
    secara otomatis tanpa butuh kata kunci 'bandingkan'.
    """
    text_lower = text.lower()
    entities = {}

    # --- STRATEGI BARU: Prioritaskan pencarian pola perbandingan ---
    # Cek apakah ini perbandingan provinsi
    prov_match = re.search(r"provinsi\s+([\w\s./'-]+?)\s+(?:dan|vs)\s+([\w\s./'-]+)", text_lower)
    if prov_match:
        entities['provinsi'] = [prov_match.group(1).strip(), prov_match.group(2).strip()]
    
    # Cek apakah ini perbandingan desa
    desa_match = re.search(r"desa\s+([\w\s./'-]+?)\s+(?:dan|vs)\s+([\w\s./'-]+)", text_lower)
    if desa_match:
        entities['desa'] = [desa_match.group(1).strip(), desa_match.group(2).strip()]
    
    # Jika salah satu perbandingan berhasil, cari juga tahun jika ada, lalu selesai.
    if entities:
        tahun_match = re.search(r"\b(\d{4})\b", text_lower)
        if tahun_match:
            entities['tahun'] = [tahun_match.group(1)]
        return entities

    # --- FALLBACK: Jika tidak ada pola perbandingan, jalankan pencarian normal ---
    patterns = {
        'desa':      r"\bdesa\s+([\w\s./'-]+?)(?=\s+dan|\s+dengan|\s+tahun|\s+\d{4}|$)",
        'kecamatan': r"\bkecamatan\s+([\w\s./'-]+?)(?=\s+dan|\s+dengan|\s+tahun|\s+\d{4}|$)",
        'kabupaten': r"\bkabupaten\s+([\w\s./'-]+?)(?=\s+dan|\s+dengan|\s+tahun|\s+\d{4}|$)",
        'provinsi':  r"\bprovinsi\s+([\w\s./'-]+?)(?=\s+dan|\s+dengan|\s+tahun|\s+\d{4}|$)",
        'tahun':     r"\b(\d{4})\b"
    }

    for name, pattern in patterns.items():
        found = re.findall(pattern, text_lower)
        if found:
            entities[name] = [item.strip() for item in found if item.strip()]
            
    return entities

# def handle_function_call(tag, function_name, user_input, entities):
#     is_projection = "proyeksi" in tag
#     narrator = narrative_proj if is_projection else narrative
#     narrator_args = entities.copy()
#     for key, value in narrator_args.items():
#         if isinstance(value, list) and len(value) == 1: narrator_args[key] = value[0]
#     narrator_args['user_input'] = user_input
    
#     try:
#         plotter_func = getattr(plotter, function_name)
        
#         # === PERBAIKAN DIMULAI DI SINI ===
#         plotter_args = {}
#         # Cek jika ini adalah fungsi perbandingan, siapkan argumennya secara khusus
#         if "bandingkan" in tag:
#             entity_type = "desa" if "desa" in tag else "provinsi"
#             items = entities.get(entity_type, [])
#             if len(items) < 2:
#                 return {"warning": f"Harap sebutkan dua nama {entity_type} untuk dibandingkan."}
#             item1, item2 = items[:2]

#             # Siapkan argumen spesifik yang dibutuhkan oleh fungsi plotter perbandingan
#             if entity_type == "desa":
#                 plotter_args = {"desa1": item1, "desa2": item2}
#             else: # provinsi
#                 plotter_args = {"provinsi1": item1, "provinsi2": item2}
        
#         # Jika bukan fungsi perbandingan, siapkan argumen secara generik (logika Anda yang sudah ada)
#         else:
#             plotter_args = {key: val for key, val in narrator_args.items() if key in plotter_func.__code__.co_varnames}
        
#         # Panggil fungsi plotter SATU KALI dengan argumen yang sudah disiapkan
#         result = plotter_func(**plotter_args, return_df=True)
#         # === PERBAIKAN SELESAI ===
        
#         # Proses hasil untuk narasi dan tampilan (kode Anda yang sudah ada)
#         if not result or (isinstance(result, tuple) and result[0] is None):
#             return {"warning": "Maaf, data tidak ditemukan untuk permintaan Anda."}

#         fig, df, trend = (None, None, None)
#         if isinstance(result, tuple):
#             fig = result[0]
#             if len(result) > 1: df = result[1]
#             if len(result) > 2: trend = result[2]
#         elif isinstance(result, pd.DataFrame): # Untuk ranking
#             df = result
#         else: # Hanya figure
#             fig = result

#         # Siapkan argumen untuk narator (kode Anda yang sudah ada)
#         if df is not None: narrator_args['df'] = df
#         if trend is not None: narrator_args['trend'] = trend
        
#         # Untuk perbandingan, pastikan argumennya ada untuk narator
#         if "bandingkan" in tag:
#             entity_type = "desa" if "desa" in tag else "provinsi"
#             items = entities.get(entity_type, [])
#             if len(items) >= 2:
#                 narrator_args.update({f'{entity_type}1': items[0], f'{entity_type}2': items[1]})

#         narration_text = narrator.generate_narrative(tag, **narrator_args)
        
#         final_result = {}
#         if fig: final_result['figure'] = fig
#         if df is not None and "ranking" in tag: final_result['dataframe'] = df
#         if narration_text: final_result['narration'] = narration_text
        
#         return final_result

#     except Exception as e:
#         import traceback
#         st.error(f"Terjadi kesalahan saat memproses '{function_name}': {e}")
#         st.error(traceback.format_exc()) # Aktifkan untuk debug lebih detail
#         return {"warning": "Maaf, terjadi kesalahan teknis saat membuat visualisasi."}

# GANTI SELURUH FUNGSI ANDA DENGAN VERSI FINAL DAN LENGKAP INI

def handle_function_call(tag, function_name, user_input, entities):
    """
    Handler pusat dengan alur logika yang sudah diperbaiki, bersih, dan terstruktur.
    Pola: Kategorikan -> Siapkan Argumen -> Jalankan -> Proses Hasil.
    """
    is_projection = "proyeksi" in tag
    narrator = narrative_proj if is_projection else narrative
    
    narrator_args = entities.copy()
    for key, value in narrator_args.items():
        if isinstance(value, list) and len(value) == 1:
            narrator_args[key] = value[0]
    narrator_args['user_input'] = user_input
    
    try:
        plotter_func = getattr(plotter, function_name)
        
        # --- KATEGORI 1: PERBANDINGAN ---
        if "bandingkan" in tag:
            entity_type = "desa" if "desa" in tag else "provinsi"
            items = entities.get(entity_type, [])
            if len(items) < 2: return {"warning": f"Harap sebutkan dua nama {entity_type} untuk dibandingkan."}
            item1, item2 = items[:2]
            
            plotter_args = {"desa1": item1, "desa2": item2} if entity_type == "desa" else {"provinsi1": item1, "provinsi2": item2}
            fig, df = plotter_func(**plotter_args, return_df=True)
            narrator_args.update(plotter_args)

        # --- KATEGORI 2: RANKING (Punya alur & pemanggilan khusus) ---
        elif "ranking" in tag:
            # Panggilan fungsi ranking TIDAK menggunakan return_df
            df_rank = plotter_func(user_input) 
            if df_rank is None or df_rank.empty: return {"warning": "Data untuk ranking tidak ditemukan."}
            
            narrator_args['df'] = df_rank
            narration_text = narrator.generate_narrative(tag, **narrator_args)
            return {"dataframe": df_rank, "narration": narration_text}

        # --- KATEGORI 3: PETA ---
        elif "peta" in tag:
            if 'tahun' in tag:
                tahun = narrator_args.get('tahun')
                if not tahun: return {"warning": "Mohon sebutkan tahun untuk menampilkan peta."}
                fig, r_max, r_min, p_max, p_min = plotter_func(int(tahun), return_regions=True)
            else: # Peta tren nasional
                fig, r_max, r_min, p_max, p_min = plotter_func(return_regions=True)
            
            if fig is None: return {"warning": "Data untuk membuat peta tidak ditemukan."}
            narrator_args.update({'region_max': r_max, 'region_min': r_min, 'prov_max': p_max, 'prov_min': p_min})
            narration_text = narrator.generate_narrative(tag, **narrator_args)
            return {"figure": fig, "narration": narration_text}

        # --- KATEGORI 4 & 5: SEMUA PLOT LAINNYA ---
        else:
            plotter_args = {key: val for key, val in narrator_args.items() if key in plotter_func.__code__.co_varnames}
            result_tuple = plotter_func(**plotter_args, return_df=True)
            
            if not result_tuple or result_tuple[0] is None:
                return {"warning": "Maaf, data tidak ditemukan untuk permintaan Anda."}

            fig, df = result_tuple[0], result_tuple[1]
            if len(result_tuple) > 2: narrator_args['trend'] = result_tuple[2]
        
        # --- Proses Hasil dan Narasi (Untuk semua kasus yang belum return) ---
        narrator_args['df'] = df
        narration_text = narrator.generate_narrative(tag, **narrator_args)
        
        return {"figure": fig, "narration": narration_text}

    except Exception as e:
        import traceback
        st.error(f"Terjadi kesalahan saat memproses '{function_name}': {e}")
        st.error(traceback.format_exc())
        return {"warning": "Maaf, terjadi kesalahan teknis saat membuat visualisasi."}

def stream_response(text, delay=0.02):
    """Displays response text with a typing effect."""
    # Split by space to yield word by word for a smoother effect
    for word in text.split():
        yield word + " "
        time.sleep(delay)

# =============================================================================
# 3. STREAMLIT USER INTERFACE
# =============================================================================

# Load resources once at the start
tokenizer, model, intents, id2label = load_resources()

st.title("🤖 Chatbot SAMUDRA-AI 🌊")
st.markdown("Tanyakan apa saja tentang tinggi muka laut (TML), proyeksi, atau kondisi per wilayah!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for controls
with st.sidebar:
    st.header("Kontrol")
    if st.button("🪑 Mulai Percakapan Baru"):
        st.session_state.messages = []
        st.rerun()

# Display chat history from session_state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Render content based on its type
        if msg["type"] == "text":
            st.markdown(msg["content"])
        elif msg["type"] == "chart":
            # Plotly charts are JSON serializable and can be stored in session state
            st.plotly_chart(msg["content"], use_container_width=True)
        elif msg["type"] == "dataframe":
            # DataFrames can also be stored directly
            st.dataframe(msg["content"], hide_index=True, use_container_width=True)

# Main chat input
if user_input := st.chat_input("Contoh: tren tml desa siomeda"):
    # Add and display user message
    st.session_state.messages.append({"role": "user", "type": "text", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process and display assistant response
    with st.chat_message("assistant"):
        tag, confidence = predict_intent(user_input, tokenizer, model, id2label)

        if confidence > 0.6:
            response_found = False
            for intent in intents['intents']:
                if intent['tag'] == tag:
                    # -- Step 1: Handle Text Response --
                    response_text = random.choice(intent['responses'])
                    st.write_stream(stream_response(response_text))
                    st.session_state.messages.append({"role": "assistant", "type": "text", "content": response_text})

                    # -- Step 2: Handle Function Call (if any) --
                    function_name = intent.get("function")
                    if function_name:
                        # 1. Inisialisasi progress bar dengan teks yang Anda inginkan
                        progress_text = "Mohon tunggu sebentar, permintaan anda sedang diproses..."
                        progress_bar = st.progress(0, text=progress_text)

                        # 2. Jalankan fungsi dan simulasikan progres
                        entities = extract_entities(user_input)
                        progress_bar.progress(33, text=progress_text)
                        time.sleep(0.5) # Jeda kecil untuk efek visual

                        result = handle_function_call(tag, function_name, user_input, entities)
                        progress_bar.progress(66, text=progress_text)
                        time.sleep(0.5) # Jeda kecil untuk efek visual
                        
                        # 3. Selesaikan progress bar sebelum menampilkan hasil
                        progress_bar.progress(100, text="Selesai!")
                        
                        # 4. Tampilkan semua hasil dari handler
                        if "warning" in result:
                            st.warning(result["warning"])
                        if "figure" in result:
                            st.plotly_chart(result["figure"], use_container_width=True)
                            st.session_state.messages.append({"role": "assistant", "type": "chart", "content": result["figure"]})
                        if "dataframe" in result:
                            st.dataframe(result["dataframe"], hide_index=True, use_container_width=True)
                            st.session_state.messages.append({"role": "assistant", "type": "dataframe", "content": result["dataframe"]})
                        if "narration" in result:
                            st.info(result["narration"])
                            st.session_state.messages.append({"role": "assistant", "type": "text", "content": result["narration"]})
                            
                        # 5. Hapus progress bar setelah selesai
                        progress_bar.empty()
                    
                    response_found = True
                    break
            
            if not response_found:
                 st.warning("Terjadi kesalahan: Tag dikenali, tetapi tidak ada definisi intent yang cocok di `samudra.json`.")

        else:
            # Fallback response for low confidence
            fallback_text = f"Maaf, saya kurang yakin memahami maksud Anda (keyakinan: {confidence:.0%}). Mohon coba tanyakan dengan lebih spesifik, misalnya 'tren tml nasional' atau 'bandingkan provinsi jawa timur dan jawa barat'."
            st.warning(fallback_text)
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": fallback_text})