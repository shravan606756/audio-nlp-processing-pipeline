import streamlit as st
from src.youtube import download_audio
from src.transcribe import transcribe_audio
from src.summarize import summarize_text

# Page Config 
st.set_page_config(
    page_title="Podcast Summarizer",
    layout="wide"
)

# Header
st.markdown(
    """
    <h1 style='text-align:center;'>Podcast Summarizer</h1>
    <p style='text-align:center; font-size:18px;'>
    Upload audio or paste a YouTube link to generate transcript and summary.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# Layout Columns
left_col, right_col = st.columns([1, 2])

with left_col:

    st.subheader("Input")

    option = st.radio(
        "Choose input type",
        ["Upload Audio", "YouTube Link"]
    )

    # Upload Audio
    if option == "Upload Audio":

        uploaded_file = st.file_uploader(
            "Upload podcast audio",
            type=["mp3", "wav", "m4a", "webm"]
        )

        if uploaded_file:

            file_path = f"data/audio/{uploaded_file.name}"

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success("File uploaded successfully")

            with st.spinner("Transcribing audio..."):
                text = transcribe_audio(file_path)

            with st.spinner("Generating summary..."):
                summary = summarize_text(text)

            st.session_state["transcript"] = text
            st.session_state["summary"] = summary

    # YouTube Input
    if option == "YouTube Link":

        url = st.text_input("Paste YouTube URL")

        if st.button("Process Podcast"):

            if url.strip() == "":
                st.warning("Enter a valid URL.")
            else:

                with st.spinner("Downloading audio..."):
                    audio_path = download_audio(url)

                with st.spinner("Transcribing audio..."):
                    text = transcribe_audio(audio_path)

                with st.spinner("Generating summary..."):
                    summary = summarize_text(text)

                st.session_state["transcript"] = text
                st.session_state["summary"] = summary




# Output Section
with right_col:

    st.subheader("Results")

    if "transcript" in st.session_state:

        tab1, tab2 = st.tabs(["Transcript", "Summary"])

        with tab1:
            st.write(st.session_state["transcript"])

        with tab2:
            st.write(st.session_state["summary"])

    else:
        st.info("Process a podcast to see results here.")
