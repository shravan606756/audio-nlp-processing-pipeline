# Audio NLP Processing Pipeline

A modular, end-to-end pipeline for automated speech transcription, semantic text segmentation, abstractive summarization via transformer inference, and retrieval-augmented generation (RAG) over audio content sourced from YouTube URLs or local file uploads.

---

## Table of Contents

- [System Architecture Overview](#system-architecture-overview)
- [Module Breakdown](#module-breakdown)
  - [Audio Ingestion and Extraction](#1-audio-ingestion-and-extraction)
  - [Speech Recognition Engine](#2-speech-recognition-engine)
  - [Text Segmentation and Chunking](#3-text-segmentation-and-chunking)
  - [Summarization Pipeline](#4-summarization-pipeline)
  - [Retrieval-Augmented Generation](#5-retrieval-augmented-generation-rag)
- [Data Flow Diagram](#data-flow-diagram)
- [Comparative Model Analysis](#comparative-model-analysis)
- [Technology Stack and Dependencies](#technology-stack-and-dependencies)
- [Environment Setup and Installation](#environment-setup-and-installation)
- [Configuration Reference](#configuration-reference)
- [Telemetry and Metrics](#telemetry-and-metrics)

---

## System Architecture Overview

The system is organized into four discrete layers: **Ingestion**, **Processing**, **Modeling**, and **Presentation**. Each layer is stateless with respect to the others — data flows downstream via well-defined buffers and serialized text objects. The Streamlit presentation layer maintains isolated session state per model to enable side-by-side comparison of BART and T5 inference outputs.

```
+-------------------------------------------------------------------------------------+
|                          PRESENTATION LAYER (Streamlit)                             |
|  Session State [BART]  |  Session State [T5]  |  RAG Query Interface                |
+-------------|--------------------|-----------------------|---------------------------+
              |                    |                       |
              v                    v                       v
+-------------------------------------------------------------------------------------+
|                             DISPATCH / ORCHESTRATION                                |
|          Validates source type, routes to ingestion module, manages state           |
+-------------------------------------------------------------------------------------+
              |
     +--------+--------+
     |                 |
     v                 v
+----------+     +------------+
| yt-dlp   |     | Binary     |
| Extractor|     | File Input |
+----------+     +------------+
     |                 |
     +--------+--------+
              |
              v
+-----------------------------+
|   Audio Multiplexer /       |
|   Format Normalization      |
|   (16kHz mono WAV target)   |
+-----------------------------+
              |
              v
+-----------------------------+
|  Whisper Transcription      |
|  Engine (base variant)      |
|  Mel-spectrogram + Attn     |
+-----------------------------+
              |
              v
+-----------------------------+
|    Raw Text Transcript      |
|    (UTF-8 string buffer)    |
+-----------------------------+
         |         |
         |         +----------------------------------+
         v                                            v
+------------------+                      +---------------------+
|  Segmentation /  |                      |  FAISS Vector Index |
|  Chunking Module |                      |  (RAG Subsystem)    |
+------------------+                      +---------------------+
         |
    +----+----+
    |         |
    v         v
+-------+ +-------+
| BART  | | T5    |
| large | | base  |
| CNN   | | 220M  |
| 406M  | |       |
+-------+ +-------+
    |         |
    v         v
+-----------------------------+
|   Structured Summary Output |
|   + Compression Metrics     |
+-----------------------------+
```

---

## Module Breakdown

### 1. Audio Ingestion and Extraction

The ingestion module normalizes two distinct input pathways into a unified audio buffer suitable for downstream Whisper inference.

**YouTube URL Path:**  
The `yt-dlp` extractor resolves the URL, fetches stream metadata, and downloads the highest-quality audio track. The output is post-processed via `ffmpeg` to normalize the sample rate to 16 kHz mono PCM, which is the required input format for Whisper.

**Local Upload Path:**  
Binary audio data submitted via the Streamlit file uploader is buffered in memory and passed through the same normalization step.

```
+--------------------+           +--------------------+
|   YouTube URL      |           |  Local File Upload |
|   (HTTP input)     |           |  (.mp3/.wav/.m4a)  |
+--------+-----------+           +---------+----------+
         |                                 |
         v                                 v
+--------+-----------+           +---------+----------+
|  URL Validation    |           |   MIME Type        |
|  + Sanitization    |           |   Detection        |
+--------+-----------+           +---------+----------+
         |                                 |
         v                                 |
+--------+-----------+                     |
|  yt-dlp Core       |                     |
|  - Format selector |                     |
|  - Metadata fetch  |                     |
|  - Stream download |                     |
|  - Temp file cache |                     |
+--------+-----------+                     |
         |                                 |
         +----------------+----------------+
                          |
                          v
              +-----------+----------+
              |  Audio Multiplexer   |
              |  ffmpeg normalization|
              |  -> 16kHz, mono, WAV |
              +-----------+----------+
                          |
                          v
              +-----------+----------+
              |  Byte Buffer /       |
              |  NumPy Float32 Array |
              |  (Whisper-compatible)|
              +----------------------+
```

**Key parameters:**
- Output format: 16000 Hz sample rate, mono channel, float32 PCM
- Temporary cache path: configurable via environment variable `AUDIO_CACHE_DIR`
- Supported input containers: `.mp3`, `.mp4`, `.wav`, `.m4a`, `.webm`, `.ogg`

---

### 2. Speech Recognition Engine

Speech-to-text inference is performed by **OpenAI Whisper (base variant)**. The engine operates on fixed-length 30-second audio windows and uses a hybrid encoder-decoder architecture based on the transformer attention mechanism.

```
+---------------------------+
|  Float32 PCM Audio Buffer |
|  (16kHz, mono)            |
+-----------+---------------+
            |
            v
+-----------+---------------+
|  Log-Mel Spectrogram      |
|  Extraction               |
|  - FFT window: 25ms       |
|  - Hop length: 10ms       |
|  - n_mels: 80             |
+-----------+---------------+
            |
            v
+-----------+---------------+
|  Convolutional Feature    |
|  Encoder (2 x Conv1D)     |
|  + GELU activations       |
+-----------+---------------+
            |
            v
+-----------+---------------+
|  Transformer Encoder      |
|  - 6 attention heads      |
|  - 512-dim hidden states  |
|  - Positional embedding   |
+-----------+---------------+
            |
            v
+-----------+---------------+
|  Autoregressive Decoder   |
|  - Cross-attention over   |
|    encoder output         |
|  - Greedy / beam search   |
|  - BPE tokenizer (50k)    |
+-----------+---------------+
            |
            v
+-----------+---------------+
|  UTF-8 Text Transcript    |
|  (with punctuation,       |
|   capitalization,         |
|   timestamp tokens opt.)  |
+---------------------------+
```

The base model contains approximately 74M parameters and supports multilingual transcription. Language detection is performed on the first 30-second window via log-probability distribution over the language token vocabulary.

---

### 3. Text Segmentation and Chunking

Transformer summarization models impose a maximum context window constraint (typically 1024 tokens for BART, 512 tokens for T5). The segmentation module breaks the raw transcript into overlapping chunks that respect these limits while preserving semantic boundaries.

```
+-----------------------------------+
|  Raw Transcript                   |
|  (variable-length UTF-8 string)   |
+-----------------+-----------------+
                  |
                  v
+-----------------+-----------------+
|  Sentence Boundary Detection      |
|  (spaCy / rule-based tokenizer)   |
+-----------------+-----------------+
                  |
                  v
+-----------------+-----------------+
|  Tokenization via HuggingFace     |
|  AutoTokenizer for target model   |
|  -> token ID sequences            |
+-----------------+-----------------+
                  |
                  v
+-----------------+-----------------+
|  Sliding Window Chunker           |
|  - max_length: model-dependent    |
|    BART: 1024 tokens              |
|    T5:    512 tokens              |
|  - stride / overlap: configurable |
|  - Boundary-aware split           |
|    (avoids mid-sentence cuts)     |
+-----------------+-----------------+
                  |
         +--------+--------+
         |        |        |
         v        v        v
      +-----+  +-----+  +-----+
      | Seg | | Seg |  | Seg |
      |  1  |  |  2  |  |  N  |
      +-----+  +-----+  +-----+
         |        |        |
         +--------+--------+
                  |
                  v
+------------------+----------------+
|  Chunk Iterator -> Inference Hub  |
+-----------------------------------+
```

Chunk metadata includes token count, character offset range, and sentence boundary indices to support post-hoc alignment of summaries back to source segments.

---

### 4. Summarization Pipeline

Each text chunk is independently passed through the selected transformer model. Per-chunk summaries are concatenated and optionally subjected to a second-pass summarization if the combined output exceeds the model's input limit.

```
+------------------------------------------+
|  Segmented Chunk [i]                     |
|  (token-bounded, boundary-aligned)       |
+--------------+---------------------------+
               |
      +--------+--------+
      |                 |
      v                 v
+-----+------+   +------+------+
|  BART-large |   |  T5-base    |
|  -CNN       |   |             |
|  406M params|   |  220M params|
|             |   |             |
|  Encoder:   |   |  Encoder:   |
|  12 layers  |   |  6 layers   |
|  1024 hidden|   |  768 hidden |
|             |   |             |
|  Decoder:   |   |  Decoder:   |
|  12 layers  |   |  6 layers   |
|  cross-attn |   |  cross-attn |
|             |   |             |
|  Beam search|   |  Greedy /   |
|  num_beams=4|   |  beam=2     |
|             |   |             |
|  max_new_   |   |  max_new_   |
|  tokens=150 |   |  tokens=100 |
+-----+------+   +------+------+
      |                 |
      v                 v
+-----+------+   +------+------+
| Chunk Sum. |   | Chunk Sum.  |
|  BART [i]  |   |   T5 [i]   |
+-----+------+   +------+------+
      |                 |
      v                 v
+-----+------+   +------+------+
| Concatenate|   | Concatenate |
| all chunks |   | all chunks  |
+-----+------+   +------+------+
      |                 |
      v                 v
+-----+------+   +------+------+
| Final BART |   | Final T5   |
| Summary    |   | Summary    |
| output     |   | output     |
+------------+   +------------+
      |                 |
      +--------+--------+
               |
               v
+-----------------------------+
|  Compression Metrics        |
|  - Input char count         |
|  - Output char count        |
|  - Compression ratio (%)    |
|  - Inference latency (s)    |
|  - Sentence count delta     |
+-----------------------------+
```

**BART-large-CNN:** Fine-tuned on CNN/DailyMail news summarization. Favors extractive-leaning abstractive output with high lexical fidelity. Compression ratio: 60-75%.

**T5-base:** Fine-tuned on C4 with summarization prefix prompting ("summarize: ..."). Produces highly abstractive, compressed output. Compression ratio: 85-95%.

---

### 5. Retrieval-Augmented Generation (RAG)

The RAG subsystem indexes the full transcript into a FAISS vector store and supports natural language querying. Relevant passages are retrieved by cosine similarity and injected as context into a generative model for answer synthesis.

```
                    INDEXING PHASE (run once per transcript)
+--------------------------------------------+
|  Full Transcript Buffer                    |
+-------------------+------------------------+
                    |
                    v
+-------------------+------------------------+
|  LangChain RecursiveCharacterTextSplitter  |
|  - chunk_size: 500 chars                   |
|  - chunk_overlap: 50 chars                 |
+-------------------+------------------------+
                    |
          +---------+---------+
          |         |         |
          v         v         v
       +------+  +------+  +------+
       | Doc  |  | Doc  |  | Doc  |
       |  1   |  |  2   |  |  N   |
       +--+---+  +--+---+  +--+---+
          |         |         |
          v         v         v
+--------------------------------------------+
|  Sentence Transformers Embedding Model     |
|  (all-MiniLM-L6-v2 or equivalent)         |
|  Output: 384-dim dense float32 vectors     |
+-------------------+------------------------+
                    |
                    v
+-------------------+------------------------+
|  FAISS IndexFlatIP                         |
|  (Inner product / cosine similarity)       |
|  In-memory flat index                      |
+--------------------------------------------+


                    QUERY PHASE (per user query)
+--------------------------------------------+
|  User Natural Language Query               |
+-------------------+------------------------+
                    |
                    v
+-------------------+------------------------+
|  Query Embedding (same encoder model)      |
|  Output: 384-dim float32 query vector      |
+-------------------+------------------------+
                    |
                    v
+-------------------+------------------------+
|  FAISS Similarity Search                   |
|  - top_k: 3-5 nearest neighbors           |
|  - Distance: inner product (normalized)    |
|  - Threshold filtering (optional)          |
+-------------------+------------------------+
                    |
                    v
+-------------------+------------------------+
|  Context Construction                      |
|  - Retrieved passages concatenated         |
|  - Prompt template injection               |
|  - Source offset metadata attached         |
+-------------------+------------------------+
                    |
                    v
+-------------------+------------------------+
|  Generative Answer Synthesis               |
|  (T5 or BART in QA mode)                  |
|  Input: [context] + [question]             |
|  Output: free-form answer string           |
+-------------------+------------------------+
                    |
                    v
+-------------------+------------------------+
|  Final Q&A Response                        |
|  + Source passage attribution              |
+--------------------------------------------+
```

---

## Data Flow Diagram

The following represents the complete end-to-end data flow from raw media input to structured output across all subsystems.

```
Raw Media Input
(YouTube URL or Binary Upload)
           |
           v
    [Validation Layer]
     URL parsing / MIME
           |
           v
   [yt-dlp / File Buffer]
     Audio stream download
     or in-memory read
           |
           v
   [Format Normalization]
     ffmpeg -> 16kHz mono
     float32 PCM array
           |
           v
   [Whisper STT Engine]
     Mel spectrogram
     Encoder-decoder attn
     BPE tokenization
           |
           v
   [Raw UTF-8 Transcript]
     Full text string
     (variable length)
           |
     +-----+-----+
     |           |
     v           v
[Chunker]   [Text Splitter]
Token-aware  Char-overlap
segmentation  for FAISS
     |           |
     v           v
[BART / T5] [Embedder]
Inference   Dense vectors
     |           |
     v           v
[Summaries] [FAISS Index]
+ metrics        |
                 v
            [User Query]
                 |
                 v
           [Similarity Search]
           top-k retrieval
                 |
                 v
           [QA Generator]
           context injection
                 |
                 v
           [Q&A Response]
```

---

## Comparative Model Analysis

| Property | BART-large-CNN | T5-base |
|---|---|---|
| Parameters | 406M | 220M |
| Architecture | Seq2Seq (BERT encoder + GPT decoder) | Unified text-to-text transformer |
| Pre-training | Denoising autoencoder on books + Wikipedia | C4 corpus (masked span prediction) |
| Fine-tuning dataset | CNN / DailyMail | CNN / DailyMail + multi-task mixture |
| Max input tokens | 1024 | 512 |
| Max output tokens | ~150 (configurable) | ~100 (configurable) |
| Compression ratio | 60-75% | 85-95% |
| Output style | Extractive-leaning abstractive | Highly abstractive, aggressive compression |
| Inference latency (CPU, ~500 token input) | ~8-14 seconds | ~3-6 seconds |
| Memory footprint | ~1.6 GB | ~880 MB |
| Decoding strategy | Beam search (num_beams=4) | Greedy or beam (num_beams=2) |
| Prompt format | Raw text | "summarize: " prefix |

Telemetry collected per inference run includes: character-level input/output footprint, sentence count before and after compression, wall-clock inference time, and a computed differential matrix when both models are run on the same input.

---

## Technology Stack and Dependencies

```
Application Layer
+----------------------------------+
|  Streamlit (UI framework)        |
|  FastAPI (underlying ASGI server)|
+----------------------------------+

Machine Learning / Inference
+----------------------------------+
|  PyTorch (LibTorch runtime)      |
|  HuggingFace Transformers        |
|  - BART-large-CNN                |
|  - T5-base                       |
|  HuggingFace Tokenizers          |
|  Sentence Transformers           |
+----------------------------------+

Audio Processing
+----------------------------------+
|  OpenAI Whisper (base)           |
|  gTTS (Google Text-to-Speech)    |
|  yt-dlp (YouTube extraction)     |
|  ffmpeg (format normalization)   |
+----------------------------------+

Vector Search / RAG
+----------------------------------+
|  FAISS (Facebook AI Similarity   |
|         Search)                  |
|  LangChain (text splitter,       |
|             retrieval chain)     |
+----------------------------------+

Data and Visualization
+----------------------------------+
|  Pandas (DataFrame operations)   |
|  Plotly Graph Objects            |
|    (interactive metric charts)   |
+----------------------------------+

Runtime
+----------------------------------+
|  Python 3.10+                    |
|  pip / virtualenv                |
+----------------------------------+
```

---

## Environment Setup and Installation

### Prerequisites

- Python 3.10 or higher (strictly required)
- `ffmpeg` installed and available on system `PATH`
- CUDA-capable GPU (optional; CPU inference is supported but slower)

### Step-by-step Installation

**1. Clone the repository**

```bash
git clone https://github.com/shravan606756/audio-nlp-processing-pipeline.git
cd audio-nlp-processing-pipeline
```

**2. Create and activate a virtual environment**

```bash
python3.10 -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**3. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**4. Verify ffmpeg availability**

```bash
ffmpeg -version
```

If `ffmpeg` is not installed, install via system package manager:

```bash
# Debian / Ubuntu
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows (via Chocolatey)
choco install ffmpeg
```

**5. (Optional) Verify CUDA / GPU availability**

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `True`, inference will utilize GPU automatically. Set `DEVICE=cpu` in the environment to force CPU execution.

**6. Launch the application**

```bash
streamlit run app/app.py
```

The application will be available at `http://localhost:8501` by default.

---

## Configuration Reference

Key hyperparameters can be adjusted via `config.yaml` or environment variables:

| Parameter | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `BART_MAX_INPUT_TOKENS` | `1024` | Maximum token length per chunk for BART |
| `T5_MAX_INPUT_TOKENS` | `512` | Maximum token length per chunk for T5 |
| `BART_MAX_NEW_TOKENS` | `150` | Maximum generation length for BART output |
| `T5_MAX_NEW_TOKENS` | `100` | Maximum generation length for T5 output |
| `CHUNK_OVERLAP` | `50` | Character overlap between RAG text splits |
| `RAG_TOP_K` | `4` | Number of nearest neighbor passages retrieved |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model for RAG embeddings |
| `DEVICE` | `auto` | Inference device: `auto`, `cpu`, `cuda` |
| `AUDIO_CACHE_DIR` | `/tmp/audio_cache` | Temporary directory for downloaded audio files |

---

## Telemetry and Metrics

The application captures and displays the following analytical metrics per inference session:

```
+--------------------------------------+
|  Per-Model Telemetry Record          |
|                                      |
|  - model_name       (str)            |
|  - input_char_count (int)            |
|  - output_char_count(int)            |
|  - compression_ratio(float, %)       |
|  - sentence_count_in (int)           |
|  - sentence_count_out(int)           |
|  - inference_time_s (float)          |
|  - num_chunks       (int)            |
|  - avg_tokens_per_chunk (float)      |
+--------------------------------------+

+--------------------------------------+
|  Differential Matrix (BART vs T5)    |
|                                      |
|  - delta_compression_ratio (float)   |
|  - delta_inference_time    (float)   |
|  - delta_output_chars      (int)     |
|  - delta_sentence_count    (int)     |
+--------------------------------------+
```

Metrics are rendered as interactive Plotly charts within the Streamlit interface and are also available as exportable Pandas DataFrames for downstream analysis.

---

## Project Structure

```
audio-nlp-processing-pipeline/
|
+-- app/
|   +-- app.py                  # Main Streamlit entry point
|   +-- ingestion.py            # Audio ingestion and yt-dlp wrapper
|   +-- transcription.py        # Whisper STT inference
|   +-- chunker.py              # Token-aware text segmentation
|   +-- summarizer.py           # BART and T5 summarization pipelines
|   +-- rag.py                  # FAISS indexing and RAG query engine
|   +-- metrics.py              # Telemetry collection and formatting
|
+-- config/
|   +-- config.yaml             # Hyperparameter configuration
|
+-- requirements.txt
+-- README.md
```

---

## License

See `LICENSE` for terms of use and distribution.