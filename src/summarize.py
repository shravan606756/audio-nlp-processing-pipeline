from transformers import pipeline
from src.chunking import split_text
import time

# Initialize both models (lazy loading)
_summarizers = {}

def get_summarizer(model_name="t5-small"):
    """Get or create summarizer with caching"""
    if model_name not in _summarizers:
        _summarizers[model_name] = pipeline(
            "summarization",
            model=model_name
        )
    return _summarizers[model_name]


# Detail level configurations safely balanced for T5's 512-token limit (approx. 350-380 words)
DETAIL_CONFIGS = {
    "t5-small": {
        "brief": {
            "chunk_size": 200,
            "chunk_overlap": 40,
            "chunk_max_length": 60,
            "chunk_min_length": 30,
            "final_max_length": 100,
            "final_min_length": 40,
            "second_level_threshold": 250,
        },
        "medium": {
            "chunk_size": 260,
            "chunk_overlap": 50,
            "chunk_max_length": 90,
            "chunk_min_length": 40,
            "final_max_length": 150,
            "final_min_length": 70,
            "second_level_threshold": 300,
        },
        "detailed": {
            "chunk_size": 320,  # Max safe size for 512 tokens
            "chunk_overlap": 60,
            "chunk_max_length": 130,
            "chunk_min_length": 60,
            "final_max_length": 250,
            "final_min_length": 100,
            "second_level_threshold": 350,
        }
    },
    "t5-base": {
        "brief": {
            "chunk_size": 220,
            "chunk_overlap": 50,
            "chunk_max_length": 70,
            "chunk_min_length": 30,
            "final_max_length": 120,
            "final_min_length": 50,
            "second_level_threshold": 280,
        },
        "medium": {
            "chunk_size": 280,
            "chunk_overlap": 60,
            "chunk_max_length": 110,
            "chunk_min_length": 50,
            "final_max_length": 180,
            "final_min_length": 80,
            "second_level_threshold": 320,
        },
        "detailed": {
            "chunk_size": 340, # Pushing the absolute safe limit
            "chunk_overlap": 70,
            "chunk_max_length": 150,
            "chunk_min_length": 80,
            "final_max_length": 300,
            "final_min_length": 120,
            "second_level_threshold": 360,
        }
    }
}


def summarize_chunks(chunks, summarizer, max_length, min_length):
    """Summarize a list of text chunks with dynamic length safeguards"""
    summaries = []

    for chunk in chunks:
        chunk_words = len(chunk.split())
        # Skip very short chunks
        if chunk_words < 20:
            continue
            
        # Dynamically adjust lengths to prevent HuggingFace max_length warnings
        estimated_tokens = int(chunk_words * 1.3)
        safe_max = min(max_length, max(30, estimated_tokens - 5))
        safe_min = min(min_length, max(10, safe_max - 15))
            
        result = summarizer(
            chunk,
            max_length=safe_max,
            min_length=safe_min,
            do_sample=False
        )
        summaries.append(result[0]["summary_text"])

    return summaries


def summarize_text(text, detail_level="medium", model_name="t5-small", return_metrics=False):
    """
    Summarize text with configurable detail level and model limits.
    """
    start_time = time.time()
    
    # Get summarizer
    summarizer = get_summarizer(model_name)
    
    # Get configuration for this model and detail level
    model_configs = DETAIL_CONFIGS.get(model_name, DETAIL_CONFIGS["t5-small"])
    config = model_configs.get(detail_level, model_configs["medium"])
    
    # Track metrics
    original_words = len(text.split())
    original_chars = len(text)
    
    # Level 1 — chunk transcript
    chunks = split_text(
        text, 
        max_words=config["chunk_size"], 
        overlap=config["chunk_overlap"]
    )
    
    num_chunks = len(chunks)

    level1_summaries = summarize_chunks(
        chunks,
        summarizer,
        max_length=config["chunk_max_length"],
        min_length=config["chunk_min_length"]
    )

    combined = " ".join(level1_summaries)

    # Level 2 — combine and potentially compress again
    if len(combined.split()) > config["second_level_threshold"]:
        second_chunks = split_text(
            combined, 
            max_words=config["chunk_size"], 
            overlap=config["chunk_overlap"]
        )
        second_level = summarize_chunks(
            second_chunks,
            summarizer,
            max_length=config["chunk_max_length"],
            min_length=config["chunk_min_length"]
        )
        combined = " ".join(second_level)

    combined_words = len(combined.split())

    # Final Pass Logic: Prevent 512-token crashes and avoid over-compressing detailed requests
    if detail_level == "detailed" or combined_words > 360 or combined_words <= (config["final_max_length"] * 1.2):
        # Bottleneck bypass: stitch it together cleanly instead of crushing it
        summary_raw = combined.replace(" .", ".").strip()
        summary = ". ".join([s.strip().capitalize() for s in summary_raw.split(".") if s.strip()]) + "."
    else:
        # Safe to compress one last time for brief/medium
        estimated_tokens = int(combined_words * 1.3)
        safe_max = min(config["final_max_length"], max(30, estimated_tokens - 10))
        safe_min = min(config["final_min_length"], max(10, safe_max - 20))
        
        final_result = summarizer(
            combined,
            max_length=safe_max,
            min_length=safe_min,
            do_sample=False
        )
        summary = final_result[0]["summary_text"]
    
    # Calculate metrics
    end_time = time.time()
    processing_time = end_time - start_time
    
    summary_words = len(summary.split())
    summary_chars = len(summary)
    compression_ratio = (1 - summary_words / original_words) * 100 if original_words > 0 else 0
    
    metrics = {
        "model": model_name,
        "detail_level": detail_level,
        "original_words": original_words,
        "original_chars": original_chars,
        "summary_words": summary_words,
        "summary_chars": summary_chars,
        "compression_ratio": compression_ratio,
        "processing_time": processing_time,
        "num_chunks": num_chunks
    }
    
    if return_metrics:
        return summary, metrics
    
    return summary


def compare_models(text, detail_level="medium"):
    """Generate summaries with both models and return comparison"""
    summary_small, metrics_small = summarize_text(
        text, 
        detail_level=detail_level, 
        model_name="t5-small",
        return_metrics=True
    )
    
    summary_base, metrics_base = summarize_text(
        text, 
        detail_level=detail_level, 
        model_name="t5-base",
        return_metrics=True
    )
    
    return {
        "t5-small": {
            "summary": summary_small,
            "metrics": metrics_small
        },
        "t5-base": {
            "summary": summary_base,
            "metrics": metrics_base
        }
    }