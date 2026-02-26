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
        color: var(--text-color);
    }
    .subtitle {
        text-align: center;
        color: var(--text-color);
        opacity: 0.7;
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
    .badge-small {
        background-color: #e3f2fd;
        color: #1976d2;
        border: 1px solid #1976d2;
    }
    .badge-base {
        background-color: #f3e5f5;
        color: #7b1fa2;
        border: 1px solid #7b1fa2;
    }
    .section-header {
        font-weight: 600;
        color: var(--text-color);
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
    
    # T5-small info
    with st.expander("T5-small (Fast)", expanded=False):
        st.caption("**Processing Speed**: 5-10 seconds")
        st.caption("**Model Size**: 250 MB (60M parameters)")
        st.caption("**Quality**: Good for quick summaries")
        st.caption("**Compression**: 90-97%")
    
    # T5-base info
    with st.expander("T5-base (High Quality)", expanded=False):
        st.caption("**Processing Speed**: 15-30 seconds")
        st.caption("**Model Size**: 900 MB (220M parameters)")
        st.caption("**Quality**: Excellent detail preservation")
        st.caption("**Compression**: 70-85%")
    
    st.divider()
    
    st.subheader("System Information")
    st.caption("Transcription: OpenAI Whisper (base)")
    st.caption("Summarization: T5-small / T5-base")
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

        # Show video info if URL is provided
        if url.strip() and show_video_info:
            with st.spinner("Fetching video metadata..."):
                video_info = get_video_info(url)
            
            if video_info:
                st.markdown("**Video Information**")
                st.markdown(f'<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"**Title**: {video_info['title']}")
                st.markdown(f"**Channel**: {video_info['channel']}")
                
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
                    st.markdown(f"**Duration**: {duration_str}")
                
                # Subtitle availability
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

                    # Step 2: Summarize with T5-small (fast first pass)
                    with st.spinner("Generating summary with T5-small..."):
                        summary, metrics = summarize_text(
                            text, 
                            detail_level=detail_level,
                            model_name="t5-small",
                            return_metrics=True
                        )

                    st.session_state["transcript"] = text
                    
                    # Save dedicated small variables explicitly to prevent overwriting
                    st.session_state["summary_small"] = summary
                    st.session_state["summary_metrics_small"] = metrics
                    
                    # Save current display variables
                    st.session_state["summary"] = summary
                    st.session_state["summary_metrics"] = metrics
                    st.session_state["source"] = source
                    st.session_state["current_model"] = "t5-small"
                    
                    # Clear any previous base summary from previous runs
                    if "summary_base" in st.session_state:
                        del st.session_state["summary_base"]
                    if "summary_metrics_base" in st.session_state:
                        del st.session_state["summary_metrics_base"]
                    if "comparison_data" in st.session_state:
                        del st.session_state["comparison_data"]
                    
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

            if st.button("Process", type="primary", width="stretch"):
                
                try:
                    # Ensure directory exists
                    os.makedirs("data/audio", exist_ok=True)
                    file_path = f"data/audio/{uploaded_file.name}"

                    # Save uploaded file
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.success("File uploaded successfully")

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

                    # Summarize with T5-small
                    with st.spinner("Generating summary with T5-small..."):
                        summary, metrics = summarize_text(
                            text,
                            detail_level=detail_level,
                            model_name="t5-small",
                            return_metrics=True
                        )

                    st.session_state["transcript"] = text
                    
                    # Save dedicated small variables explicitly
                    st.session_state["summary_small"] = summary
                    st.session_state["summary_metrics_small"] = metrics
                    
                    # Save current display variables
                    st.session_state["summary"] = summary
                    st.session_state["summary_metrics"] = metrics
                    st.session_state["source"] = source
                    st.session_state["current_model"] = "t5-small"
                    
                    # Clear any previous base summary
                    if "summary_base" in st.session_state:
                        del st.session_state["summary_base"]
                    if "summary_metrics_base" in st.session_state:
                        del st.session_state["summary_metrics_base"]
                    if "comparison_data" in st.session_state:
                        del st.session_state["comparison_data"]
                    
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
        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Transcript", "Analytics", "Model Comparison"])

        with tab1:
            # Current model badge
            current_model = st.session_state.get("current_model", "t5-small")
            badge_class = "badge-small" if current_model == "t5-small" else "badge-base"
            
            st.markdown(
                f'<span class="model-badge {badge_class}">{current_model.upper()}</span>',
                unsafe_allow_html=True
            )
            st.markdown('<p class="section-header">Generated Summary</p>', unsafe_allow_html=True)
            
            summary_text = st.session_state["summary"]
            st.write(summary_text)
            
            # Show metrics if available
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
            
            # Action buttons
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
                # Regenerate button logic with cached states
                if current_model == "t5-small":
                    if st.button("Regenerate with T5-base", width="stretch", type="secondary"):
                        with st.spinner("Generating higher quality summary with T5-base..."):
                            summary_base, metrics_base = summarize_text(
                                st.session_state["transcript"],
                                detail_level=detail_level,
                                model_name="t5-base",
                                return_metrics=True
                            )
                            
                            # Save explicitly to base variables
                            st.session_state["summary_base"] = summary_base
                            st.session_state["summary_metrics_base"] = metrics_base
                            
                            # Set current view
                            st.session_state["summary"] = summary_base
                            st.session_state["summary_metrics"] = metrics_base
                            st.session_state["current_model"] = "t5-base"
                            
                        st.success("Summary regenerated with T5-base model")
                        st.rerun()
                else:
                    if st.button("Switch to T5-small", width="stretch", type="secondary"):
                        # Restore instantly from our cached _small variables
                        st.session_state["summary"] = st.session_state["summary_small"]
                        st.session_state["summary_metrics"] = st.session_state["summary_metrics_small"]
                        st.session_state["current_model"] = "t5-small"
                        st.rerun()
            
            # Show comparison notice if both summaries exist
            if "summary_base" in st.session_state and current_model == "t5-small":
                st.info("Both T5-small and T5-base summaries are available. View the Model Comparison tab for detailed analysis.")

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
                st.metric("Avg words per sentence", f"{avg_words}")
            
            with col2:
                st.markdown("**Summary Metrics**")
                st.metric("Words", f"{summ_words:,}")
                st.metric("Characters", f"{summ_chars:,}")
                st.metric("Compression ratio", f"{compression_ratio:.1f}%")
                reading_time = summ_words // 150  # Average reading speed
                st.metric("Estimated reading time", f"{reading_time} min")
            
            # Visual comparison using Plotly
            st.markdown("**Length Comparison**")
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Transcript',
                x=['Word Count'],
                y=[trans_words],
                marker_color='#1f77b4',
                text=[f'{trans_words:,}'],
                textposition='auto',
            ))
            
            fig.add_trace(go.Bar(
                name='Summary',
                x=['Word Count'],
                y=[summ_words],
                marker_color='#7b1fa2',
                text=[f'{summ_words:,}'],
                textposition='auto',
            ))
            
            fig.update_layout(
                height=400,
                showlegend=True,
                yaxis_title="Words",
                template="plotly_white",
                font=dict(size=12)
            )
            
            st.plotly_chart(fig, width="stretch")

        with tab4:
            st.markdown('<p class="section-header">Model Comparison</p>', unsafe_allow_html=True)
            
            # Check if we have both summaries saved properly in session state
            has_both = "summary_base" in st.session_state and "summary_small" in st.session_state
            
            if not has_both:
                st.info("Generate a T5-base summary to enable model comparison. Click 'Regenerate with T5-base' in the Summary tab.")
                
                st.markdown("### Expected Model Characteristics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### T5-small")
                    st.markdown("""
                    - **Processing Speed**: 5-10 seconds
                    - **Model Size**: 250 MB (60M parameters)
                    - **Compression**: 90-97%
                    - **Quality**: Good for quick summaries
                    - **Use Case**: Fast iteration, quick overview
                    """)
                
                with col2:
                    st.markdown("#### T5-base")
                    st.markdown("""
                    - **Processing Speed**: 15-30 seconds
                    - **Model Size**: 900 MB (220M parameters)
                    - **Compression**: 70-85%
                    - **Quality**: Excellent detail preservation
                    - **Use Case**: High-quality analysis, detailed summaries
                    """)
                
                # Show theoretical comparison chart
                st.markdown("### Performance Comparison")
                
                comparison_df = pd.DataFrame({
                    'Metric': ['Speed (sec)', 'Quality Score', 'Detail Level', 'Model Size (MB)'],
                    'T5-small': [7, 70, 75, 250],
                    'T5-base': [22, 95, 92, 900]
                })
                
                fig = go.Figure()
                
                metrics_to_plot = ['Speed (sec)', 'Quality Score', 'Detail Level']
                
                for metric in metrics_to_plot:
                    row = comparison_df[comparison_df['Metric'] == metric]
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=['T5-small', 'T5-base'],
                        y=[row['T5-small'].values[0], row['T5-base'].values[0]],
                    ))
                
                fig.update_layout(
                    barmode='group',
                    height=400,
                    yaxis_title="Score",
                    template="plotly_white",
                    showlegend=True,
                    font=dict(size=12)
                )
                
                st.plotly_chart(fig, width="stretch")
                
            else:
                # We have both summaries - show detailed comparison
                
                # Get metrics from explicit cached states
                if "summary_metrics_small" in st.session_state and "summary_metrics_base" in st.session_state:
                    metrics_small = st.session_state["summary_metrics_small"]
                    metrics_base = st.session_state["summary_metrics_base"]
                    
                    # Side by side summaries
                    st.markdown("### Summary Comparison")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### T5-small Summary")
                        st.markdown(f'<div class="stats-box">Words: {metrics_small["summary_words"]:,} | '
                                  f'Time: {metrics_small["processing_time"]:.1f}s | '
                                  f'Compression: {metrics_small["compression_ratio"]:.1f}%</div>',
                                  unsafe_allow_html=True)
                        # Read directly from explicit _small state
                        st.write(st.session_state["summary_small"])
                    
                    with col2:
                        st.markdown("#### T5-base Summary")
                        st.markdown(f'<div class="stats-box">Words: {metrics_base["summary_words"]:,} | '
                                  f'Time: {metrics_base["processing_time"]:.1f}s | '
                                  f'Compression: {metrics_base["compression_ratio"]:.1f}%</div>',
                                  unsafe_allow_html=True)
                        st.write(st.session_state["summary_base"])
                    
                    st.divider()
                    
                    # Metrics comparison
                    st.markdown("### Detailed Metrics Comparison")
                    
                    # Create comparison dataframe
                    comp_df = pd.DataFrame({
                        'Metric': [
                            'Summary Words',
                            'Processing Time (s)',
                            'Compression Ratio (%)',
                            'Chunks Processed'
                        ],
                        'T5-small': [
                            metrics_small['summary_words'],
                            round(metrics_small['processing_time'], 1),
                            round(metrics_small['compression_ratio'], 1),
                            metrics_small['num_chunks']
                        ],
                        'T5-base': [
                            metrics_base['summary_words'],
                            round(metrics_base['processing_time'], 1),
                            round(metrics_base['compression_ratio'], 1),
                            metrics_base['num_chunks']
                        ],
                        'Difference': [
                            metrics_base['summary_words'] - metrics_small['summary_words'],
                            round(metrics_base['processing_time'] - metrics_small['processing_time'], 1),
                            round(metrics_base['compression_ratio'] - metrics_small['compression_ratio'], 1),
                            metrics_base['num_chunks'] - metrics_small['num_chunks']
                        ]
                    })
                    
                    st.dataframe(comp_df, width="stretch", hide_index=True)
                    
                    # Visual comparison charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Processing time comparison
                        fig_time = go.Figure()
                        fig_time.add_trace(go.Bar(
                            x=['T5-small', 'T5-base'],
                            y=[metrics_small['processing_time'], metrics_base['processing_time']],
                            marker_color=['#1f77b4', '#7b1fa2'],
                            text=[f"{metrics_small['processing_time']:.1f}s", 
                                  f"{metrics_base['processing_time']:.1f}s"],
                            textposition='auto',
                        ))
                        fig_time.update_layout(
                            title="Processing Time",
                            yaxis_title="Seconds",
                            height=300,
                            template="plotly_white",
                            showlegend=False,
                            font=dict(size=12)
                        )
                        st.plotly_chart(fig_time, width="stretch")
                    
                    with col2:
                        # Summary length comparison
                        fig_words = go.Figure()
                        fig_words.add_trace(go.Bar(
                            x=['T5-small', 'T5-base'],
                            y=[metrics_small['summary_words'], metrics_base['summary_words']],
                            marker_color=['#1f77b4', '#7b1fa2'],
                            text=[f"{metrics_small['summary_words']:,}", 
                                  f"{metrics_base['summary_words']:,}"],
                            textposition='auto',
                        ))
                        fig_words.update_layout(
                            title="Summary Length",
                            yaxis_title="Words",
                            height=300,
                            template="plotly_white",
                            showlegend=False,
                            font=dict(size=12)
                        )
                        st.plotly_chart(fig_words, width="stretch")
                    
                    # Compression comparison
                    fig_compression = go.Figure()
                    fig_compression.add_trace(go.Bar(
                        x=['T5-small', 'T5-base'],
                        y=[metrics_small['compression_ratio'], metrics_base['compression_ratio']],
                        marker_color=['#1f77b4', '#7b1fa2'],
                        text=[f"{metrics_small['compression_ratio']:.1f}%", 
                              f"{metrics_base['compression_ratio']:.1f}%"],
                        textposition='auto',
                    ))
                    fig_compression.update_layout(
                        title="Compression Ratio",
                        yaxis_title="Compression %",
                        height=300,
                        template="plotly_white",
                        showlegend=False,
                        font=dict(size=12)
                    )
                    st.plotly_chart(fig_compression, width="stretch")
                    
                    # Key insights
                    st.markdown("### Performance Analysis")
                    
                    time_diff = metrics_base['processing_time'] - metrics_small['processing_time']
                    word_diff = metrics_base['summary_words'] - metrics_small['summary_words']
                    compression_diff = metrics_base['compression_ratio'] - metrics_small['compression_ratio']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Speed Trade-off",
                            f"+{time_diff:.1f}s",
                            f"{(time_diff/metrics_small['processing_time']*100):.0f}% slower" if metrics_small['processing_time'] > 0 else "0% slower",
                            delta_color="inverse"
                        )
                    
                    with col2:
                        st.metric(
                            "Detail Gain",
                            f"+{word_diff:,} words",
                            f"{(word_diff/metrics_small['summary_words']*100):.0f}% more content" if metrics_small['summary_words'] > 0 else "0% more content"
                        )
                    
                    with col3:
                        if compression_diff < 0:
                            st.metric(
                                "Compression Delta",
                                f"{abs(compression_diff):.1f}%",
                                "Lower compression (more details)"
                            )
                        else:
                            st.metric(
                                "Compression Delta",
                                f"{compression_diff:.1f}%",
                                "Similar compression"
                            )

    else:
        st.info("Process a podcast or video to view results")
        
        with st.expander("Usage Instructions"):
            st.markdown("""
            ### YouTube Processing
            1. Paste a YouTube URL in the input field
            2. Click **Process** to begin extraction
            3. Review T5-small summary (5-10 seconds)
            4. Optionally click **Regenerate with T5-base** for higher quality
            5. Compare both models in the **Model Comparison** tab
            
            ### Audio File Processing
            1. Upload an audio file (MP3, WAV, M4A, WebM, OGG)
            2. Click **Process** to begin transcription
            3. Follow same steps as YouTube processing
            
            ### Model Selection
            - **T5-small**: Fast results, good for quick summaries
            - **T5-base**: Higher quality, more detailed, better accuracy
            - **Compare both**: See the difference side-by-side
            
            ### Performance Expectations
            - Videos with captions: 3-5 seconds
            - Whisper transcription: 1-5 minutes (depends on audio length)
            - T5-small summarization: 5-10 seconds
            - T5-base summarization: 15-30 seconds
            """)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>Powered by OpenAI Whisper, T5-small, T5-base, and yt-dlp</small>
    </div>
""", unsafe_allow_html=True)