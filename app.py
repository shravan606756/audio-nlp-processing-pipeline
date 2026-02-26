import streamlit as st
import os
from src.youtube import fetch_youtube_transcript, download_audio, get_video_info
from src.transcribe import transcribe_audio
from src.summarize import summarize_text

# Page Config
st.set_page_config(
    page_title="Podcast Summarizer",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f1f1f;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stats-box {
        background-color: #e8f4f8;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
        color: #856404;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Podcast Summarizer</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Automated transcript extraction and AI-powered summarization</p>',
    unsafe_allow_html=True
)

st.divider()

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    st.subheader("Display Options")
    show_video_info = st.checkbox("Show video metadata", value=True)
    show_stats = st.checkbox("Show processing statistics", value=True)
    
    st.divider()
    
    st.subheader("Summary Settings")
    detail_level = st.select_slider(
        "Summary detail level",
        options=["brief", "medium", "detailed"],
        value="medium",
        help="Brief: ~95% compression, Medium: ~85% compression, Detailed: ~70% compression"
    )
    
    st.divider()
    
    st.subheader("System Information")
    st.caption("Transcription: OpenAI Whisper (base)")
    st.caption("Summarization: T5-small")
    st.caption("YouTube Extraction: yt-dlp")
    
    st.divider()
    
    st.subheader("Performance Notes")
    st.caption("• Videos with captions: 3-5 seconds")
    st.caption("• Whisper transcription: 1-5 minutes")
    st.caption("• Summarization: 10-30 seconds")

# Main Layout
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Input")

    option = st.radio(
        "Select input type",
        ["YouTube Link", "Upload Audio"]
    )

    # -------------------- YOUTUBE INPUT --------------------
    if option == "YouTube Link":
        
        url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=..."
        )

        # Show video info if URL is provided
        if url.strip() and show_video_info:
            with st.spinner("Fetching video metadata..."):
                video_info = get_video_info(url)
            
            if video_info:
                st.markdown("**Video Information**")
                st.markdown(f'<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"**Title:** {video_info['title']}")
                st.markdown(f"**Channel:** {video_info['channel']}")
                
                # Format duration
                duration = video_info['duration']
                if duration:
                    hours = duration // 3600
                    minutes = (duration % 3600) // 60
                    seconds = duration % 60
                    if hours > 0:
                        duration_str = f"{hours}h {minutes}m {seconds}s"
                    else:
                        duration_str = f"{minutes}m {seconds}s"
                    st.markdown(f"**Duration:** {duration_str}")
                
                # Subtitle availability
                if video_info['has_manual_subs']:
                    st.markdown("**Subtitles:** Manual captions available")
                elif video_info['has_auto_subs']:
                    st.markdown("**Subtitles:** Auto-generated captions available")
                else:
                    st.markdown("**Subtitles:** None (will use Whisper)")
                
                st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Process", type="primary", use_container_width=True):

            if url.strip() == "":
                st.error("Please enter a valid URL")
            else:
                try:
                    # Step 1: Try YouTube Transcript
                    with st.spinner("Extracting transcript..."):
                        text, source_type = fetch_youtube_transcript(url)

                    if text is not None:
                        # Success with transcript
                        if source_type == "manual":
                            st.markdown(
                                '<div class="success-box">Using manual YouTube captions</div>',
                                unsafe_allow_html=True
                            )
                            source = "YouTube Captions (Manual)"
                        else:
                            st.markdown(
                                '<div class="success-box">Using auto-generated YouTube captions</div>',
                                unsafe_allow_html=True
                            )
                            source = "YouTube Captions (Auto-generated)"
                        
                        # Show stats
                        if show_stats:
                            word_count = len(text.split())
                            char_count = len(text)
                            st.markdown(
                                f'<div class="stats-box">Transcript: {word_count:,} words, {char_count:,} characters</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        # Fallback to Whisper
                        st.markdown(
                            '<div class="warning-box">No transcript available. Downloading audio for Whisper transcription...</div>',
                            unsafe_allow_html=True
                        )

                        with st.spinner("Downloading audio..."):
                            audio_path = download_audio(url)
                        
                        if not audio_path:
                            st.markdown(
                                '<div class="error-box">Failed to download audio. Please verify the URL.</div>',
                                unsafe_allow_html=True
                            )
                            st.stop()

                        with st.spinner("Transcribing audio (this may take several minutes)..."):
                            text = transcribe_audio(audio_path)

                        source = "Whisper Transcription"
                        
                        if show_stats:
                            word_count = len(text.split())
                            st.markdown(
                                f'<div class="stats-box">Transcript: {word_count:,} words</div>',
                                unsafe_allow_html=True
                            )

                    # Step 2: Summarize
                    with st.spinner("Generating summary..."):
                        summary = summarize_text(text, detail_level=detail_level)

                    st.session_state["transcript"] = text
                    st.session_state["summary"] = summary
                    st.session_state["source"] = source
                    
                    st.success("Processing complete")
                    
                except Exception as e:
                    st.markdown(
                        f'<div class="error-box">Error: {str(e)}</div>',
                        unsafe_allow_html=True
                    )


    # -------------------- AUDIO UPLOAD --------------------
    elif option == "Upload Audio":

        uploaded_file = st.file_uploader(
            "Audio file",
            type=["mp3", "wav", "m4a", "webm", "ogg"]
        )

        if uploaded_file:
            
            # Show file info
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.markdown(f'<div class="stats-box">File: {uploaded_file.name} ({file_size_mb:.2f} MB)</div>', 
                       unsafe_allow_html=True)

            if st.button("Process", type="primary", use_container_width=True):
                
                try:
                    # Ensure directory exists
                    os.makedirs("data/audio", exist_ok=True)
                    file_path = f"data/audio/{uploaded_file.name}"

                    # Save uploaded file
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.success("File uploaded")

                    # Transcribe
                    with st.spinner("Transcribing audio..."):
                        text = transcribe_audio(file_path)
                        source = "Whisper Transcription"

                    if show_stats:
                        word_count = len(text.split())
                        st.markdown(
                            f'<div class="stats-box">Transcript: {word_count:,} words</div>',
                            unsafe_allow_html=True
                        )

                    # Summarize
                    with st.spinner("Generating summary..."):
                        summary = summarize_text(text, detail_level=detail_level)

                    st.session_state["transcript"] = text
                    st.session_state["summary"] = summary
                    st.session_state["source"] = source
                    
                    st.success("Processing complete")
                    
                except Exception as e:
                    st.markdown(
                        f'<div class="error-box">Error: {str(e)}</div>',
                        unsafe_allow_html=True
                    )


# -------------------- OUTPUT SECTION --------------------
with right_col:
    st.subheader("Results")

    if "transcript" in st.session_state:

        # Source indicator
        st.markdown(
            f'<div class="stats-box">Source: {st.session_state["source"]}</div>',
            unsafe_allow_html=True
        )

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Summary", "Transcript", "Analytics"])

        with tab1:
            st.markdown("**AI-Generated Summary**")
            summary_text = st.session_state["summary"]
            st.write(summary_text)
            
            st.download_button(
                label="Download Summary",
                data=summary_text,
                file_name="summary.txt",
                mime="text/plain",
                use_container_width=True
            )

        with tab2:
            st.markdown("**Full Transcript**")
            transcript_text = st.session_state["transcript"]
            st.write(transcript_text)
            
            st.download_button(
                label="Download Transcript",
                data=transcript_text,
                file_name="transcript.txt",
                mime="text/plain",
                use_container_width=True
            )

        with tab3:
            st.markdown("**Text Analytics**")
            
            transcript = st.session_state["transcript"]
            summary = st.session_state["summary"]
            
            # Calculate metrics
            trans_words = len(transcript.split())
            trans_chars = len(transcript)
            trans_sentences = transcript.count('.') + transcript.count('!') + transcript.count('?')
            
            summ_words = len(summary.split())
            summ_chars = len(summary)
            
            compression_ratio = (1 - summ_words / trans_words) * 100 if trans_words > 0 else 0
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Transcript Metrics**")
                st.metric("Words", f"{trans_words:,}")
                st.metric("Characters", f"{trans_chars:,}")
                st.metric("Sentences", f"{trans_sentences:,}")
                avg_words = trans_words // trans_sentences if trans_sentences > 0 else 0
                st.metric("Average words per sentence", f"{avg_words}")
            
            with col2:
                st.markdown("**Summary Metrics**")
                st.metric("Words", f"{summ_words:,}")
                st.metric("Characters", f"{summ_chars:,}")
                st.metric("Compression ratio", f"{compression_ratio:.1f}%")
                reading_time = summ_words // 150  # Average reading speed
                st.metric("Estimated reading time", f"{reading_time} min")
            
            # Visual comparison
            st.markdown("**Length Comparison**")
            import pandas as pd
            
            comparison_data = pd.DataFrame({
                'Type': ['Transcript', 'Summary'],
                'Word Count': [trans_words, summ_words]
            })
            
            st.bar_chart(comparison_data.set_index('Type'))

    else:
        st.info("Process a podcast to view results")
        
        with st.expander("Usage Instructions"):
            st.markdown("""
            **YouTube Processing:**
            1. Paste a YouTube URL in the input field
            2. Click "Process" to begin extraction
            3. System will attempt to use existing captions first
            4. Falls back to Whisper transcription if captions unavailable
            
            **Audio File Processing:**
            1. Upload an audio file (MP3, WAV, M4A, WebM, OGG)
            2. Click "Process" to begin transcription
            3. Whisper will transcribe the audio
            
            **Performance Notes:**
            - YouTube videos with captions process in 3-5 seconds
            - Whisper transcription takes 1-5 minutes depending on audio length
            - Summarization typically completes in 10-30 seconds
            """)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>Powered by OpenAI Whisper, T5-small, and yt-dlp</small>
    </div>
""", unsafe_allow_html=True)