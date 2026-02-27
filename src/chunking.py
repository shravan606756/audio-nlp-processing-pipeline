import re

def split_text(text, max_words=200, overlap=40):
    """
    Intelligently chunks text by sentence boundaries to preserve context,
    while strictly respecting maximum word limits.
    """
    # Fallback for empty text
    if not text or not text.strip():
        return []

    # 1. Split text into sentences using regex
    # Looks for punctuation (. ! ?) followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    chunks = []
    current_chunk_sentences = []
    current_word_count = 0
    
    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)
        
        # SAFETY FALLBACK: For YouTube auto-captions without punctuation.
        # If a single "sentence" is larger than our max limit, we MUST slice it by words.
        if word_count > max_words:
            # Save any existing sentences we've collected
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_word_count = 0
            
            # Brutally slice the massive unpunctuated block to prevent token crashes
            i = 0
            while i < word_count:
                chunk_slice = words[i:i + max_words]
                chunks.append(" ".join(chunk_slice))
                i += max_words - overlap
            
            # Set up the last slice as the start of the next potential chunk
            if chunks:
                last_chunk_words = chunks[-1].split()
                current_chunk_sentences = [" ".join(last_chunk_words[-overlap:])] if overlap > 0 else []
                current_word_count = len(current_chunk_sentences[0].split()) if current_chunk_sentences else 0
            continue

        # NORMAL FLOW: Add sentence to chunk if it fits
        if current_word_count + word_count <= max_words:
            current_chunk_sentences.append(sentence)
            current_word_count += word_count
        else:
            # Chunk is full! Save it.
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
            
            # Create the overlap for the next chunk using FULL SENTENCES
            overlap_sentences = []
            overlap_words = 0
            
            # Work backwards through the current chunk to grab sentences for overlap
            for s in reversed(current_chunk_sentences):
                s_words = len(s.split())
                if overlap_words + s_words <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_words += s_words
                else:
                    break
            
            # If a single sentence was larger than the overlap, just keep that last sentence
            if not overlap_sentences and current_chunk_sentences:
                overlap_sentences = [current_chunk_sentences[-1]]
                
            # Start the new chunk with the overlap + the new sentence
            current_chunk_sentences = overlap_sentences + [sentence]
            current_word_count = sum(len(s.split()) for s in current_chunk_sentences)
                
    # Append the final chunk if anything is leftover
    if current_chunk_sentences:
        final_chunk = " ".join(current_chunk_sentences)
        # Avoid appending a duplicate if the text ended perfectly on a chunk boundary
        if not chunks or final_chunk != chunks[-1]:
            chunks.append(final_chunk)
            
    return chunks