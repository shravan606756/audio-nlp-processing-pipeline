from transformers import pipeline
from src.chunking import split_text

summarizer = pipeline(
    "summarization",
    model="t5-small"
)

def summarize_level(texts):
    summaries = []

    for t in texts:
        result = summarizer(
            t,
            max_length=120,
            min_length=30,
            do_sample=False
        )
        summaries.append(result[0]["summary_text"])

    return summaries


def summarize_text(text):

    # Level 1
    chunks = split_text(text, max_words=200, overlap=40)
    level1 = summarize_level(chunks)

    # Level 2
    grouped = split_text(" ".join(level1), max_words=300, overlap=50)
    level2 = summarize_level(grouped)

    # Final
    final_summary = summarize_level([" ".join(level2)])[0]

    return final_summary