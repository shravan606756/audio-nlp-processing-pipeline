from transformers import pipeline
from src.chunking import split_text
import time

_summarizers = {}

def get_summarizer(model_name="bart-large-cnn"):
    """Get or create summarizer with caching"""
    if model_name not in _summarizers:
        model_map = {
            "bart-large-cnn": "facebook/bart-large-cnn",
            "t5-base": "t5-base",
        }
        
        full_model_name = model_map.get(model_name, model_name)
        print(f"Loading model: {full_model_name}...")
        
        _summarizers[model_name] = pipeline(
            "summarization",
            model=full_model_name,
            device=-1  # CPU mode
        )
        print(f"Model {model_name} loaded successfully")
    return _summarizers[model_name]

# Detail configs balanced for token limits (BART: 1024 tokens = ~750 words max)
DETAIL_CONFIGS = {
    "bart-large-cnn": {
        "brief": {"chunk_size": 600, "chunk_overlap": 100, "chunk_max_length": 150, "chunk_min_length": 50, "second_level_threshold": 800},
        "medium": {"chunk_size": 450, "chunk_overlap": 80, "chunk_max_length": 180, "chunk_min_length": 80, "second_level_threshold": 9999},
        "detailed": {"chunk_size": 350, "chunk_overlap": 60, "chunk_max_length": 250, "chunk_min_length": 120, "second_level_threshold": 9999}
    },
    "t5-base": {
        "brief": {"chunk_size": 220, "chunk_overlap": 50, "chunk_max_length": 70, "chunk_min_length": 30, "second_level_threshold": 280},
        "medium": {"chunk_size": 180, "chunk_overlap": 40, "chunk_max_length": 110, "chunk_min_length": 50, "second_level_threshold": 450},
        "detailed": {"chunk_size": 120, "chunk_overlap": 30, "chunk_max_length": 150, "chunk_min_length": 80, "second_level_threshold": 9999}
    }
}

def cleanup_summary(text):
    """Smooths out stitched text into a cohesive essay."""
    text = text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
    text = " ".join(text.split()) 
    sentences = [s.strip().capitalize() for s in text.split(".") if s.strip()]
    if not sentences: return text
    return ". ".join(sentences) + "."

def summarize_chunks(chunks, summarizer, max_length, min_length, model_name):
    summaries = []
    for chunk in chunks:
        chunk_words = len(chunk.split())
        if chunk_words < 20: continue
            
        estimated_tokens = int(chunk_words * 1.3)
        safe_max = min(max_length, max(30, estimated_tokens - 5))
        safe_min = min(min_length, max(10, safe_max - 20))
            
        result = summarizer(chunk, max_length=safe_max, min_length=safe_min, do_sample=False)
        summaries.append(result[0]["summary_text"])
    return summaries

def summarize_text(text, detail_level="medium", model_name="bart-large-cnn", return_metrics=False):
    start_time = time.time()
    summarizer = get_summarizer(model_name)
    
    config = DETAIL_CONFIGS.get(model_name, DETAIL_CONFIGS["bart-large-cnn"])[detail_level]
    original_words = len(text.split())
    
    # LEVEL 1: Linear Chunking
    chunks = split_text(text, max_words=config["chunk_size"], overlap=config["chunk_overlap"])
    num_chunks = len(chunks)

    level1_summaries = summarize_chunks(
        chunks, summarizer,
        max_length=config["chunk_max_length"],
        min_length=config["chunk_min_length"],
        model_name=model_name
    )

    combined_linear = " ".join(level1_summaries)
    combined_words = len(combined_linear.split())

    # LEVEL 2 & ROUTING
    if detail_level == "detailed":
        summary = cleanup_summary(combined_linear)
        
    elif detail_level == "medium":
        if combined_words > config["second_level_threshold"]:
            second_chunks = split_text(combined_linear, max_words=400, overlap=50)
            level2 = summarize_chunks(second_chunks, summarizer, max_length=150, min_length=60, model_name=model_name)
            summary = cleanup_summary(" ".join(level2))
        else:
            summary = cleanup_summary(combined_linear)
            
    else:
        # Brief
        if combined_words > config["second_level_threshold"]:
            second_chunks = split_text(combined_linear, max_words=400, overlap=50)
            level2 = summarize_chunks(second_chunks, summarizer, max_length=100, min_length=40, model_name=model_name)
            combined_linear = " ".join(level2)
            
        estimated_tokens = int(len(combined_linear.split()) * 1.3)
        token_limit = 1000 if "bart" in model_name.lower() else 450
        
        if estimated_tokens > token_limit:
            summary = cleanup_summary(combined_linear)
        else:
            safe_max = min(200 if "bart" in model_name else 120, max(30, estimated_tokens - 10))
            safe_min = min(80 if "bart" in model_name else 50, max(10, safe_max - 20))
            final_result = summarizer(combined_linear, max_length=safe_max, min_length=safe_min, do_sample=False)
            summary = cleanup_summary(final_result[0]["summary_text"])
    
    summary_words = len(summary.split())
    metrics = {
        "model": model_name,
        "detail_level": detail_level,
        "original_words": original_words,
        "original_chars": len(text),
        "summary_words": summary_words,
        "summary_chars": len(summary),
        "compression_ratio": (1 - summary_words / original_words) * 100 if original_words > 0 else 0,
        "processing_time": time.time() - start_time,
        "num_chunks": num_chunks
    }
    
    return (summary, metrics) if return_metrics else summary