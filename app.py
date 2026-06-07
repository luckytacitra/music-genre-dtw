import streamlit as st
import librosa
import librosa.display
import numpy as np
import pickle
import matplotlib.pyplot as plt

from dtaidistance import dtw

st.set_page_config(
    page_title="Music Genre Recognition",
    layout="centered"
)

st.title("🎵 Music Genre Recognition")
st.write(
    "Klasifikasi genre musik menggunakan Mel Frequency Cepstral Coefficients (MFCC) dan Dynamic Time Warping (DTW)."
)

with open("database.pkl", "rb") as f:
    database = pickle.load(f)

uploaded_file = st.file_uploader(
    "Upload file musik",
    type=["wav", "mp3"]
)

if uploaded_file is not None:

    # ==========================
    # LOAD AUDIO
    # ==========================

    y, sr = librosa.load(
        uploaded_file,
        duration=3
    )

    st.subheader("📄 Audio Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "File Name",
            uploaded_file.name
        )

    with col2:
        st.metric(
            "File Size",
            f"{uploaded_file.size/1024/1024:.2f} MB"
        )

    with col3:
        st.metric(
            "Sampling Rate",
            sr
        )

    # ==========================
    # AUDIO PLAYER
    # ==========================

    st.subheader("▶ Audio Player")
    st.audio(uploaded_file)

    # ==========================
    # WAVEFORM
    # ==========================

    import matplotlib.pyplot as plt
    import librosa.display

    st.subheader("📈 Waveform")

    fig, ax = plt.subplots(
        figsize=(10,3)
    )

    librosa.display.waveshow(
        y,
        sr=sr,
        ax=ax
    )

    ax.set_title(
        "Audio Waveform"
    )

    st.pyplot(fig)

    # ==========================
    # MFCC
    # ==========================

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    st.subheader("🎼 MFCC")

    fig2, ax2 = plt.subplots(
        figsize=(10,4)
    )

    img = librosa.display.specshow(
        mfcc,
        x_axis='time',
        ax=ax2
    )

    plt.colorbar(
        img,
        ax=ax2
    )

    ax2.set_title(
        "MFCC Feature"
    )

    st.pyplot(fig2)

    # ==========================
    # PREPARE DTW
    # ==========================

    test_mfcc = np.mean(
        mfcc,
        axis=1
    )

    best_distance = float("inf")
    best_genre = ""

    all_result = []

    for item in database:

        dist = dtw.distance(
            test_mfcc,
            item["mfcc"]
        )

        all_result.append(
            (
                item["genre"],
                dist
            )
        )

        if dist < best_distance:

            best_distance = dist
            best_genre = item["genre"]

    # ==========================
    # RESULT
    # ==========================

    st.markdown("---")

    st.subheader("🏆 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Genre",
            best_genre.upper()
        )

    with col2:

        st.metric(
            "DTW Distance",
            round(
                best_distance,
                2
            )
        )

    # ==========================
    # TOP 3
    # ==========================

    all_result.sort(
        key=lambda x: x[1]
    )

    st.subheader(
        "🥇 Top 3 Closest Genres"
    )

    medal = ["🥇", "🥈", "🥉"]

for i in range(3):

    st.write(
        f"{medal[i]} {all_result[i][0].upper()}   ({all_result[i][1]:.2f})"
    )
