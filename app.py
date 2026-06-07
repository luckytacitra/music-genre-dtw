import streamlit as st
import librosa
import numpy as np
import pickle

from dtaidistance import dtw

st.set_page_config(
    page_title="Music Genre Recognition",
    layout="centered"
)

st.title("🎵 Music Genre Recognition")
st.write("Dynamic Time Warping (DTW) + MFCC")

with open("database.pkl", "rb") as f:
    database = pickle.load(f)

uploaded_file = st.file_uploader(
    "Upload file musik (.wav)",
    type=["wav"]
)

if uploaded_file is not None:

    y, sr = librosa.load(
        uploaded_file,
        duration=3
    )

    test_mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    test_mfcc = np.mean(
        test_mfcc,
        axis=1
    )

    best_distance = float("inf")
    best_genre = ""

    for item in database:

        dist = dtw.distance(
            test_mfcc,
            item["mfcc"]
        )

        if dist < best_distance:
            best_distance = dist
            best_genre = item["genre"]

    st.success(
        f"Prediction : {best_genre.upper()}"
    )

    st.write(
        f"DTW Distance : {best_distance:.4f}"
    )
