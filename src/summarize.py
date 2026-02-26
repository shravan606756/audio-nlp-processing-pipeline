from transformers import pipeline
from src.chunking import split_text

# Simple, stable summarizer
summarizer = pipeline(
    "summarization",
    model="t5-small"
)


def summarize_chunks(chunks):
    summaries = []

    for chunk in chunks:
        result = summarizer(
            chunk,
            max_length=140,
            min_length=40,
            do_sample=False
        )
        summaries.append(result[0]["summary_text"])

    return summaries


def summarize_text(text):

    # Level 1 — chunk transcript
    chunks = split_text(text, max_words=200, overlap=40)

    level1_summaries = summarize_chunks(chunks)

    # Level 2 — combine and compress again
    combined = " ".join(level1_summaries)

    # If combined text is still large, compress once more
    if len(combined.split()) > 300:
        second_chunks = split_text(combined, max_words=200, overlap=40)
        second_level = summarize_chunks(second_chunks)
        combined = " ".join(second_level)

    final_result = summarizer(
        combined,
        max_length=180,
        min_length=60,
        do_sample=False
    )

    return final_result[0]["summary_text"]