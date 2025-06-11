import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier

# --- KONFIGURASI DAN PEMETAAN (MAPPING) ---
# Hardcode pemetaan dari teks ke angka sesuai dengan LabelEncoder di notebook
# Ini penting agar model menerima input yang benar
island_mapping = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}
sex_mapping = {'FEMALE': 0, 'MALE': 1}

# Pemetaan hasil prediksi (angka) kembali ke nama spesies
species_mapping = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

# --- FUNGSI UNTUK MEMUAT MODEL ---
# Menggunakan cache agar model tidak perlu dimuat ulang setiap kali ada interaksi
@st.cache_resource
def load_model(path):
    """Memuat model machine learning dari file .pkl"""
    model = joblib.load(path)
    return model

# --- TAMPILAN APLIKASI WEB (UI) ---
st.set_page_config(page_title="Prediksi Spesies Penguin", page_icon="🐧", layout="centered")
st.title("🐧 Prediksi Spesies Penguin")
st.write(
    "Aplikasi ini menggunakan model **K-Nearest Neighbors (KNN)** untuk memprediksi "
    "spesies penguin berdasarkan atribut fisiknya. Silakan masukkan data di bawah ini."
)
st.markdown("---")

# Memuat model
model = load_model('penguin_model.pkl')

# --- FORM INPUT DATA DARI PENGGUNA ---
with st.form("penguin_input_form"):
    st.subheader("Masukkan Atribut Penguin:")
    
    # Membuat dua kolom untuk tampilan yang lebih rapi
    col1, col2 = st.columns(2)
    
    with col1:
        island_input = st.selectbox("Pulau (Island)", options=list(island_mapping.keys()))
        sex_input = st.selectbox("Jenis Kelamin (Sex)", options=list(sex_mapping.keys()))
        culmen_length = st.number_input("Panjang Paruh (Culmen Length, mm)", min_value=30.0, max_value=60.0, value=44.0, step=0.1)
        culmen_depth = st.number_input("Kedalaman Paruh (Culmen Depth, mm)", min_value=13.0, max_value=22.0, value=17.0, step=0.1)
        
    with col2:
        flipper_length = st.number_input("Panjang Sirip (Flipper Length, mm)", min_value=170.0, max_value=235.0, value=200.0, step=1.0)
        body_mass = st.number_input("Massa Tubuh (Body Mass, g)", min_value=2700.0, max_value=6300.0, value=4200.0, step=50.0)
        delta_15_n = st.number_input("Delta 15 N (o/oo)", min_value=7.0, max_value=10.0, value=8.7, step=0.1)
        delta_13_c = st.number_input("Delta 13 C (o/oo)", min_value=-28.0, max_value=-23.0, value=-25.6, step=0.1)

    # Tombol untuk mengirim form
    submit_button = st.form_submit_button(label="Lakukan Prediksi")

# --- PROSES PREDIKSI SETELAH TOMBOL DITEKAN ---
if submit_button:
    # 1. Mengubah input teks menjadi angka menggunakan mapping
    island_encoded = island_mapping[island_input]
    sex_encoded = sex_mapping[sex_input]
    
    # 2. Membuat array input sesuai urutan fitur saat training
    # Urutannya harus sama persis dengan urutan kolom di X_final
    # Urutan kolom: ['Island', 'Culmen Length (mm)', 'Culmen Depth (mm)', 
    #                'Flipper Length (mm)', 'Body Mass (g)', 'Sex', 
    #                'Delta 15 N (o/oo)', 'Delta 13 C (o/oo)']
    input_data = np.array([[
        island_encoded, 
        culmen_length, 
        culmen_depth, 
        flipper_length, 
        body_mass, 
        sex_encoded,
        delta_15_n,
        delta_13_c
    ]])
    
    # 3. Melakukan prediksi
    prediction_encoded = model.predict(input_data)
    
    # 4. Mengambil probabilitas prediksi
    prediction_proba = model.predict_proba(input_data)
    
    # 5. Mengubah hasil prediksi (angka) kembali ke nama spesies
    prediction_species = species_mapping[prediction_encoded[0]]
    
    # Menampilkan hasil
    st.markdown("---")
    st.subheader("Hasil Prediksi:")
    
    if prediction_species == 'Adelie':
        st.success(f"Spesies Penguin adalah **{prediction_species}** 🐧")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/Adelie_Penguin_composite.jpg/800px-Adelie_Penguin_composite.jpg")
    elif prediction_species == 'Chinstrap':
        st.success(f"Spesies Penguin adalah **{prediction_species}** 🐧")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Chinstrap_penguin_%28Pygoscelis_antarctica%29_2.jpg/800px-Chinstrap_penguin_%28Pygoscelis_antarctica%29_2.jpg")
    else: # Gentoo
        st.success(f"Spesies Penguin adalah **{prediction_species}** 🐧")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Gentoo_Penguin_undertakes_a_long_march_to_the_sea.jpg/800px-Gentoo_Penguin_undertakes_a_long_march_to_the_sea.jpg")

    st.write("Tingkat keyakinan model:")
    st.write(f"- Adelie: **{prediction_proba[0][0]:.2%}**")
    st.write(f"- Chinstrap: **{prediction_proba[0][1]:.2%}**")
    st.write(f"- Gentoo: **{prediction_proba[0][2]:.2%}**")

