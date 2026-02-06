import streamlit as st
from src.youtube import download_audio
from src.transcribe import transcribe_audio
from src.summarize import summarize_text

st.title("Podcast Summarizer")

option = st.radio("Choose input type", ["Upload Audio", "YouTube Link"])

# =========================
# Upload Audio Option
# =========================

if option == "Upload Audio":

    uploaded_file = st.file_uploader(
        "Upload podcast audio",
        type=["mp3", "wav", "m4a", "webm"]
    )

    if uploaded_file:

        # save uploaded file locally
        file_path = f"data/audio/{uploaded_file.name}"

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("File uploaded successfully!")

        with st.spinner("Transcribing audio..."):
            text = transcribe_audio(file_path)

        st.subheader("Transcript")
        st.write(text)

        with st.spinner("Generating summary..."):
            summary = summarize_text(text)

        st.subheader("Summary")
        st.write(summary)


# =========================
# YouTube Option
# =========================

if option == "YouTube Link":

    url = st.text_input("Paste YouTube URL", key="youtube_url")

    if st.button("Download Audio"):

        if url.strip() == "":
            st.warning("Please enter a YouTube URL.")
        else:

            with st.spinner("Downloading audio from YouTube..."):
                audio_path = download_audio(url)

            st.success(f"Audio downloaded: {audio_path}")

            with st.spinner("Transcribing audio..."):
                text = transcribe_audio(audio_path)

            st.subheader("Transcript")
            st.write(text)

            with st.spinner("Generating summary..."):
                summary = summarize_text(text)

            st.subheader("Summary")
            st.write(summary)
