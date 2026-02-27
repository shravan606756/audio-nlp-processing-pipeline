import streamlit as st
import os
from src.youtube import fetch_youtube_transcript, download_audio, get_video_info
from src.transcribe import transcribe_audio
from src.summarize import summarize_text
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page Config
st.set_page_config(
    page_title="Podcast Summarizer Pro",
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
        color: var(--text-color); /* Fixed black-on-black */
    }
    .subtitle {
        text-align: center;
        color: var(--text-color); /* Fixed black-on-black */
        opacity: 0.8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #1f1f1f;
    }
    .stats-box {
        background-color: #e8f4f8;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        color: #1f1f1f;
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
    .model-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        font-weight: 600;
        margin-right: 0.5rem;
        font-family: monospace;
    }
    .badge-bart {
        background-color: #e3f2fd;
        color: #1976d2;
        border: 1px solid #1976d2;
    }
    .badge-t5 {
        background-color: #f3e5f5;
        color: #7b1fa2;
        border: 1px solid #7b1fa2;
    }
    .section-header {
        font-weight: 600;
        color: var(--text-color); /* Fixed black-on-black */
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Podcast Summarizer Pro</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">AI-Powered Transcript Extraction and Multi-Model Summarization</p>',
    unsafe_allow_html=True
)

st.divider()

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Display Options")
    show_video_info = st.checkbox("Show video metadata", value=True)
    show_stats = st.checkbox("Show processing statistics", value=True)
    
    st.divider()
    
    st.subheader("Summary Configuration")
    detail_level = st.select_slider(
        "Detail level",
        options=["brief", "medium", "detailed"],
        value="medium",
        help="Brief: ~95% compression | Medium: ~85% compression | Detailed: ~70% compression"
    )
    
    st.divider()
    
    st.subheader("Model Information")
    
    with st.expander("BART-large-CNN (Detailed)", expanded=False):
        st.caption("**Processing Speed**: 20-40 seconds")
        st.caption("**Model Size**: 1.6 GB (406M parameters)")
        st.caption("**Quality**: Excellent for detailed summaries")
        st.caption("**Compression**: 60-75%")
    
    with st.expander("T5-base (Comparison)", expanded=False):
        st.caption("**Processing Speed**: 15-30 seconds")
        st.caption("**Model Size**: 900 MB (220M parameters)")
        st.caption("**Quality**: Good but aggressive compression")
        st.caption("**Compression**: 85-95%")
    
    st.divider()
    
    st.subheader("System Information")
    st.caption("Transcription: OpenAI Whisper (base)")
    st.caption("Summarization: BART-large-CNN / T5-base")
    st.caption("Video Extraction: yt-dlp")

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

        if url.strip() and show_video_info:
            with st.spinner("Fetching video metadata..."):
                video_info = get_video_info(url)
            
            if video_info:
                st.markdown("**Video Information**")
                st.markdown(f'<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"**Title**: {video_info['title']}")
                st.markdown(f"**Channel**: {video_info['channel']}")
                
                duration = video_info['duration']
                if duration:
                    hours = duration // 3600
                    minutes = (duration % 3600) // 60
                    seconds = duration % 60
                    if hours > 0:
                        duration_str = f"{hours}h {minutes}m {seconds}s"
                    else:
                        duration_str = f"{minutes}m {seconds}s"
                    st.markdown(f"**Duration**: {duration_str}")
                
                if video_info['has_manual_subs']:
                    st.markdown("**Subtitles**: Manual captions available")
                elif video_info['has_auto_subs']:
                    st.markdown("**Subtitles**: Auto-generated captions available")
                else:
                    st.markdown("**Subtitles**: None (will use Whisper transcription)")
                
                st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Process", type="primary", width="stretch"):

            if url.strip() == "":
                st.error("Please enter a valid URL")
            else:
                try:
                    with st.spinner("Extracting transcript..."):
                        text, source_type = fetch_youtube_transcript(url)

                    if text is not None:
                        if source_type == "manual":
                            st.markdown('<div class="success-box">Using manual YouTube captions</div>', unsafe_allow_html=True)
                            source = "YouTube Captions (Manual)"
                        else:
                            st.markdown('<div class="success-box">Using auto-generated YouTube captions</div>', unsafe_allow_html=True)
                            source = "YouTube Captions (Auto-generated)"
                        
                        if show_stats:
                            word_count = len(text.split())
                            char_count = len(text)
                            st.markdown(f'<div class="stats-box">Transcript: {word_count:,} words, {char_count:,} characters</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">No transcript available. Downloading audio for Whisper transcription...</div>', unsafe_allow_html=True)

                        with st.spinner("Downloading audio..."):
                            audio_path = download_audio(url)
                        
                        if not audio_path:
                            st.markdown('<div class="error-box">Failed to download audio. Please verify the URL.</div>', unsafe_allow_html=True)
                            st.stop()

                        with st.spinner("Transcribing audio (this may take several minutes)..."):
                            text = transcribe_audio(audio_path)

                        source = "Whisper Transcription"
                        
                        if show_stats:
                            word_count = len(text.split())
                            st.markdown(f'<div class="stats-box">Transcript: {word_count:,} words</div>', unsafe_allow_html=True)

                    with st.spinner("Generating summary with BART-large-CNN..."):
                        summary, metrics = summarize_text(
                            text, 
                            detail_level=detail_level,
                            model_name="bart-large-cnn",
                            return_metrics=True
                        )

                    st.session_state["transcript"] = text
                    
                    # FIX: Explicitly save BART metrics so they don't get lost
                    st.session_state["summary_bart"] = summary
                    st.session_state["summary_metrics_bart"] = metrics
                    
                    st.session_state["summary"] = summary
                    st.session_state["summary_metrics"] = metrics
                    st.session_state["source"] = source
                    st.session_state["current_model"] = "bart-large-cnn"
                    
                    if "summary_base" in st.session_state:
                        del st.session_state["summary_base"]
                    if "summary_metrics_base" in st.session_state:
                        del st.session_state["summary_metrics_base"]
                    
                    st.success("Processing complete")
                    
                except Exception as e:
                    st.markdown(f'<div class="error-box">Error: {str(e)}</div>', unsafe_allow_html=True)


    # -------------------- AUDIO UPLOAD --------------------
    elif option == "Upload Audio":

        uploaded_file = st.file_uploader("Audio file", type=["mp3", "wav", "m4a", "webm", "ogg"])

        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.markdown(f'<div class="stats-box">File: {uploaded_file.name} ({file_size_mb:.2f} MB)</div>', unsafe_allow_html=True)

            if st.button("Process", type="primary", width="stretch"):
                try:
                    os.makedirs("data/audio", exist_ok=True)
                    file_path = f"data/audio/{uploaded_file.name}"

                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.success("File uploaded successfully")

                    with st.spinner("Transcribing audio..."):
                        text = transcribe_audio(file_path)
                        source = "Whisper Transcription"

                    if show_stats:
                        word_count = len(text.split())
                        st.markdown(f'<div class="stats-box">Transcript: {word_count:,} words</div>', unsafe_allow_html=True)

                    with st.spinner("Generating summary with BART-large-CNN..."):
                        summary, metrics = summarize_text(
                            text,
                            detail_level=detail_level,
                            model_name="bart-large-cnn",
                            return_metrics=True
                        )

                    st.session_state["transcript"] = text
                    
                    # FIX: Explicitly save BART metrics
                    st.session_state["summary_bart"] = summary
                    st.session_state["summary_metrics_bart"] = metrics
                    
                    st.session_state["summary"] = summary
                    st.session_state["summary_metrics"] = metrics
                    st.session_state["source"] = source
                    st.session_state["current_model"] = "bart-large-cnn"
                    
                    if "summary_base" in st.session_state:
                        del st.session_state["summary_base"]
                    if "summary_metrics_base" in st.session_state:
                        del st.session_state["summary_metrics_base"]
                    
                    st.success("Processing complete")
                    
                except Exception as e:
                    st.markdown(f'<div class="error-box">Error: {str(e)}</div>', unsafe_allow_html=True)

# -------------------- OUTPUT SECTION --------------------
with right_col:
    st.subheader("Results")

    if "transcript" in st.session_state:

        st.markdown(f'<div class="stats-box">Source: {st.session_state["source"]}</div>', unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Transcript", "Analytics", "Model Comparison"])

        with tab1:
            current_model = st.session_state.get("current_model", "bart-large-cnn")
            badge_class = "badge-bart" if current_model == "bart-large-cnn" else "badge-t5"
            
            st.markdown(f'<span class="model-badge {badge_class}">{current_model.upper()}</span>', unsafe_allow_html=True)
            st.markdown('<p class="section-header">Generated Summary</p>', unsafe_allow_html=True)
            
            summary_text = st.session_state["summary"]
            st.write(summary_text)
            
            if "summary_metrics" in st.session_state:
                metrics = st.session_state["summary_metrics"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Words", f"{metrics['summary_words']:,}")
                with col2:
                    st.metric("Compression", f"{metrics['compression_ratio']:.1f}%")
                with col3:
                    st.metric("Processing Time", f"{metrics['processing_time']:.1f}s")
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="Download Summary",
                    data=summary_text,
                    file_name="summary.txt",
                    mime="text/plain",
                    width="stretch"
                )
            
            with col2:
                if current_model == "bart-large-cnn":
                    if st.button("Regenerate with T5-base", width="stretch", type="secondary"):
                        with st.spinner("Generating summary with T5-base..."):
                            summary_base, metrics_base = summarize_text(
                                st.session_state["transcript"],
                                detail_level=detail_level,
                                model_name="t5-base",
                                return_metrics=True
                            )
                            
                            # FIX: Save T5 metrics cleanly to their own variable
                            st.session_state["summary_base"] = summary_base
                            st.session_state["summary_metrics_base"] = metrics_base
                            
                            st.session_state["summary"] = summary_base
                            st.session_state["summary_metrics"] = metrics_base
                            st.session_state["current_model"] = "t5-base"
                            
                        st.success("Summary regenerated with T5-base model")
                        st.rerun()
                else:
                    if st.button("Switch to BART-large-CNN", width="stretch", type="secondary"):
                        # FIX: Pull cleanly from our saved variables
                        st.session_state["summary"] = st.session_state["summary_bart"]
                        st.session_state["summary_metrics"] = st.session_state["summary_metrics_bart"]
                        st.session_state["current_model"] = "bart-large-cnn"
                        st.rerun()
            
            if "summary_base" in st.session_state and current_model == "bart-large-cnn":
                st.info("Both BART-large-CNN and T5-base summaries are available. View the Model Comparison tab for detailed analysis.")

        with tab2:
            st.markdown('<p class="section-header">Full Transcript</p>', unsafe_allow_html=True)
            transcript_text = st.session_state["transcript"]
            st.write(transcript_text)
            
            st.download_button(
                label="Download Transcript",
                data=transcript_text,
                file_name="transcript.txt",
                mime="text/plain",
                width="stretch"
            )

        with tab3:
            st.markdown('<p class="section-header">Text Analytics</p>', unsafe_allow_html=True)
            
            transcript = st.session_state["transcript"]
            summary = st.session_state["summary"]
            
            trans_words = len(transcript.split())
            trans_chars = len(transcript)
            trans_sentences = transcript.count('.') + transcript.count('!') + transcript.count('?')
            
            summ_words = len(summary.split())
            summ_chars = len(summary)
            
            compression_ratio = (1 - summ_words / trans_words) * 100 if trans_words > 0 else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Transcript Metrics**")
                st.metric("Words", f"{trans_words:,}")
                st.metric("Characters", f"{trans_chars:,}")
                st.metric("Sentences", f"{trans_sentences:,}")
                avg_words = trans_words // trans_sentences if trans_sentences > 0 else 0
                st.metric("Avg words per sentence", f"{avg_words}")
            
            with col2:
                st.markdown("**Summary Metrics**")
                st.metric("Words", f"{summ_words:,}")
                st.metric("Characters", f"{summ_chars:,}")
                st.metric("Compression ratio", f"{compression_ratio:.1f}%")
                reading_time = summ_words // 150 
                st.metric("Estimated reading time", f"{reading_time} min")
            
            st.markdown("**Length Comparison**")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Transcript', x=['Word Count'], y=[trans_words], marker_color='#1f77b4', text=[f'{trans_words:,}'], textposition='auto'
            ))
            fig.add_trace(go.Bar(
                name='Summary', x=['Word Count'], y=[summ_words], marker_color='#7b1fa2', text=[f'{summ_words:,}'], textposition='auto'
            ))
            fig.update_layout(height=400, showlegend=True, yaxis_title="Words", template="plotly_white", font=dict(size=12))
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.markdown('<p class="section-header">Model Comparison</p>', unsafe_allow_html=True)
            
            has_both = "summary_base" in st.session_state and "summary_bart" in st.session_state
            
            if not has_both:
                st.info("Generate a T5-base summary to enable model comparison. Click 'Regenerate with T5-base' in the Summary tab.")
                
                st.markdown("### Expected Model Characteristics")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### BART-large-CNN")
                    st.markdown("- **Processing Speed**: 20-40 seconds\n- **Model Size**: 1.6 GB\n- **Quality**: Excellent for detailed summaries")
                with col2:
                    st.markdown("#### T5-base")
                    st.markdown("- **Processing Speed**: 15-30 seconds\n- **Model Size**: 900 MB\n- **Quality**: Good but more aggressive compression")
            else:
                # FIX: Pull strictly from the explicitly saved variables to prevent 0 differences
                if "summary_metrics_bart" in st.session_state and "summary_metrics_base" in st.session_state:
                    metrics_bart = st.session_state["summary_metrics_bart"]
                    metrics_t5 = st.session_state["summary_metrics_base"]
                    
                    st.markdown("### Summary Comparison")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### BART-large-CNN Summary")
                        st.markdown(f'<div class="stats-box">Words: {metrics_bart["summary_words"]:,} | '
                                  f'Time: {metrics_bart["processing_time"]:.1f}s | '
                                  f'Compression: {metrics_bart["compression_ratio"]:.1f}%</div>',
                                  unsafe_allow_html=True)
                        st.write(st.session_state["summary_bart"])
                    
                    with col2:
                        st.markdown("#### T5-base Summary")
                        st.markdown(f'<div class="stats-box">Words: {metrics_t5["summary_words"]:,} | '
                                  f'Time: {metrics_t5["processing_time"]:.1f}s | '
                                  f'Compression: {metrics_t5["compression_ratio"]:.1f}%</div>',
                                  unsafe_allow_html=True)
                        st.write(st.session_state["summary_base"])
                    
                    st.divider()
                    st.markdown("### Detailed Metrics Comparison")
                    
                    comp_df = pd.DataFrame({
                        'Metric': ['Summary Words', 'Processing Time (s)', 'Compression Ratio (%)', 'Chunks Processed'],
                        'BART-large-CNN': [metrics_bart['summary_words'], round(metrics_bart['processing_time'], 1), round(metrics_bart['compression_ratio'], 1), metrics_bart['num_chunks']],
                        'T5-base': [metrics_t5['summary_words'], round(metrics_t5['processing_time'], 1), round(metrics_t5['compression_ratio'], 1), metrics_t5['num_chunks']],
                        'Difference': [
                            metrics_t5['summary_words'] - metrics_bart['summary_words'],
                            round(metrics_t5['processing_time'] - metrics_bart['processing_time'], 1),
                            round(metrics_t5['compression_ratio'] - metrics_bart['compression_ratio'], 1),
                            metrics_t5['num_chunks'] - metrics_bart['num_chunks']
                        ]
                    })
                    
                    st.dataframe(comp_df, width="stretch", hide_index=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_time = go.Figure()
                        fig_time.add_trace(go.Bar(
                            x=['BART-large-CNN', 'T5-base'], y=[metrics_bart['processing_time'], metrics_t5['processing_time']], marker_color=['#1f77b4', '#7b1fa2'],
                            text=[f"{metrics_bart['processing_time']:.1f}s", f"{metrics_t5['processing_time']:.1f}s"], textposition='auto',
                        ))
                        fig_time.update_layout(title="Processing Time", yaxis_title="Seconds", height=300, template="plotly_white", showlegend=False, font=dict(size=12))
                        st.plotly_chart(fig_time, use_container_width=True)
                    
                    with col2:
                        fig_words = go.Figure()
                        fig_words.add_trace(go.Bar(
                            x=['BART-large-CNN', 'T5-base'], y=[metrics_bart['summary_words'], metrics_t5['summary_words']], marker_color=['#1f77b4', '#7b1fa2'],
                            text=[f"{metrics_bart['summary_words']:,}", f"{metrics_t5['summary_words']:,}"], textposition='auto',
                        ))
                        fig_words.update_layout(title="Summary Length", yaxis_title="Words", height=300, template="plotly_white", showlegend=False, font=dict(size=12))
                        st.plotly_chart(fig_words, use_container_width=True)
                    
                    fig_compression = go.Figure()
                    fig_compression.add_trace(go.Bar(
                        x=['BART-large-CNN', 'T5-base'], y=[metrics_bart['compression_ratio'], metrics_t5['compression_ratio']], marker_color=['#1f77b4', '#7b1fa2'],
                        text=[f"{metrics_bart['compression_ratio']:.1f}%", f"{metrics_t5['compression_ratio']:.1f}%"], textposition='auto',
                    ))
                    fig_compression.update_layout(title="Compression Ratio", yaxis_title="Compression %", height=300, template="plotly_white", showlegend=False, font=dict(size=12))
                    st.plotly_chart(fig_compression, use_container_width=True)
                    
                    st.markdown("### Performance Analysis")
                    
                    time_diff = metrics_t5['processing_time'] - metrics_bart['processing_time']
                    word_diff = metrics_bart['summary_words'] - metrics_t5['summary_words']
                    compression_diff = metrics_t5['compression_ratio'] - metrics_bart['compression_ratio']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if time_diff > 0:
                            st.metric("T5 Speed Advantage", f"{abs(time_diff):.1f}s faster", f"{(abs(time_diff)/metrics_bart['processing_time']*100):.0f}% faster")
                        else:
                            st.metric("BART Speed Advantage", f"{abs(time_diff):.1f}s faster", f"{(abs(time_diff)/metrics_t5['processing_time']*100):.0f}% faster" if metrics_t5['processing_time'] > 0 else "0%")
                    
                    with col2:
                        st.metric("BART Detail Advantage", f"+{word_diff:,} words", f"{(word_diff/metrics_t5['summary_words']*100):.0f}% more content" if metrics_t5['summary_words'] > 0 else "0%")
                    
                    with col3:
                        st.metric("Compression Delta", f"{abs(compression_diff):.1f}%", "BART preserves more detail" if compression_diff > 0 else "Similar compression")

    else:
        st.info("Process a podcast or video to view results")

st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>Powered by OpenAI Whisper, BART-large-CNN, T5-base, and yt-dlp</small>
    </div>
""", unsafe_allow_html=True)