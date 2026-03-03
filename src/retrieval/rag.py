import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.processing.chunking import split_text

_embedding_model = None


def get_embedding_model():
    """Lazy load the embedding model"""
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model loaded successfully")
    return _embedding_model


def build_vector_store(transcript, chunk_size=350, overlap=60):
    """
    Builds FAISS index from transcript.
    
    Args:
        transcript: Full transcript text
        chunk_size: Maximum words per chunk
        overlap: Word overlap between chunks
    
    Returns:
        index: FAISS index
        chunks: list of text chunks
    """
    model = get_embedding_model()

    # Split transcript into chunks
    chunks = split_text(transcript, max_words=chunk_size, overlap=overlap)
    if not chunks:
        return None, []

    print(f"Created {len(chunks)} chunks for RAG")

    # Generate embeddings
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal} vectors")

    return index, chunks


def retrieve_chunks(question, index, chunks, top_k=3):
    """
    Retrieves most relevant chunks for a question using semantic search.
    
    Args:
        question: User's question
        index: FAISS index
        chunks: List of text chunks
        top_k: Number of top chunks to retrieve
    
    Returns:
        List of retrieved text chunks
    """
    if index is None or not chunks:
        return []
    
    model = get_embedding_model()
    
    # Encode question
    question_embedding = model.encode([question], convert_to_numpy=True, show_progress_bar=False)

    # Search in FAISS index
    distances, indices = index.search(question_embedding, min(top_k, len(chunks)))

    # Retrieve chunks
    retrieved = []
    for i, dist in zip(indices[0], distances[0]):
        if i < len(chunks):
            retrieved.append({
                'text': chunks[i],
                'distance': float(dist),
                'index': int(i)
            })

    return retrieved


def generate_answer(question, index, chunks, detail_level="medium"):
    """
    Generates answer using retrieved context.
    Uses simple extractive approach for reliability.
    
    Args:
        question: User's question
        index: FAISS index
        chunks: List of text chunks
        detail_level: "brief", "medium", or "detailed"
    
    Returns:
        Answer string
    """
    # Set top_k based on detail level
    top_k_map = {
        "brief": 1,
        "medium": 2,
        "detailed": 3
    }
    top_k = top_k_map.get(detail_level, 2)
    
    # Retrieve relevant chunks
    retrieved_chunks = retrieve_chunks(question, index, chunks, top_k=top_k)

    if not retrieved_chunks:
        return "I couldn't find relevant information in the transcript to answer your question."

    # Format answer based on detail level
    if detail_level == "brief":
        # Return just the most relevant chunk
        return f"**Answer:**\n\n{retrieved_chunks[0]['text']}"
    
    elif detail_level == "medium":
        # Return top 2 chunks with context
        answer = "**Answer (based on transcript):**\n\n"
        for i, chunk_data in enumerate(retrieved_chunks, 1):
            answer += f"{chunk_data['text']}\n\n"
        return answer.strip()
    
    else:  # detailed
        # Return top 3 chunks with numbering
        answer = "**Detailed Answer (from transcript):**\n\n"
        for i, chunk_data in enumerate(retrieved_chunks, 1):
            answer += f"**Context {i}:**\n{chunk_data['text']}\n\n"
        return answer.strip()


def search_transcript(query, index, chunks, top_k=5):
    """
    Simple search function for finding specific information.
    
    Args:
        query: Search query
        index: FAISS index
        chunks: List of text chunks
        top_k: Number of results to return
    
    Returns:
        List of matching chunks with relevance scores
    """
    retrieved = retrieve_chunks(query, index, chunks, top_k=top_k)
    
    results = []
    for chunk_data in retrieved:
        results.append({
            'text': chunk_data['text'],
            'relevance': 1 / (1 + chunk_data['distance'])  # Convert distance to relevance score
        })
    
    return results