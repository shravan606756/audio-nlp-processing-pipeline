import pytest
from unittest.mock import patch, MagicMock
from src.retrieval.rag import generate_answer

def test_generate_answer_success():
    """Test that the Groq LLM generates an answer successfully when chunks are found."""
    mock_chunks = ["chunk 1", "chunk 2"]
    
    # Mock retrieve_chunks so we don't need a real FAISS index
    with patch('src.retrieval.rag.retrieve_chunks') as mock_retrieve:
        mock_retrieve.return_value = [
            {'text': 'Mocked relevant transcript context'}
        ]
        
        # Mock the OpenAI client so we don't hit the real Groq API during CI testing
        with patch('src.retrieval.rag.OpenAI') as mock_openai:
            # Setup the mocked response chain
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "This is a mock Llama 3 generated answer."
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            # Execute
            answer = generate_answer("What is the takeaway?", None, mock_chunks)
            
            # Assertions
            assert answer == "This is a mock Llama 3 generated answer."
            mock_retrieve.assert_called_once()
            mock_client.chat.completions.create.assert_called_once()

def test_generate_answer_not_found():
    """Test that the system returns 'Not found' when no relevant chunks correspond to the query."""
    with patch('src.retrieval.rag.retrieve_chunks') as mock_retrieve:
        mock_retrieve.return_value = []
        
        answer = generate_answer("Unknown topic?", None, [])
        assert answer == "Not found"

def test_generate_answer_api_error():
    """Test graceful error handling if the Groq API fails."""
    mock_chunks = ["chunk"]
    
    with patch('src.retrieval.rag.retrieve_chunks') as mock_retrieve:
        mock_retrieve.return_value = [{'text': 'Context'}]
        
        with patch('src.retrieval.rag.OpenAI') as mock_openai:
            mock_client = MagicMock()
            # Force the API call to throw an exception
            mock_client.chat.completions.create.side_effect = Exception("API rate limit exceeded")
            mock_openai.return_value = mock_client
            
            answer = generate_answer("Question?", None, mock_chunks)
            
            assert "Error generating answer using Groq API: API rate limit exceeded" in answer
