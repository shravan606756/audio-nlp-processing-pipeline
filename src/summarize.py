from transformers import pipeline
from src.chunking import split_text

summarizer = pipeline(
    "summarization",
    model="t5-small"
)

def summarize_text(text):

    chunks = split_text(text)

    summaries = []

    for chunk in chunks:

        result = summarizer(
            chunk,
            max_length=120,
            min_length=30,
            do_sample=False
        )

        summaries.append(result[0]["summary_text"])

    final_summary = " ".join(summaries)

    return final_summary
