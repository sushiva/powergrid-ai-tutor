"""
Embedding generation for converting text to vector representations.
"""

import os
from typing import List
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini

# Load environment variables from .env file
load_dotenv()


class EmbeddingManager:
    """
    Manages embedding generation and LLM configuration.
    Sets up global LlamaIndex settings for embeddings and LLM.
    """
    
    def __init__(self, 
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 llm_model: str = "models/gemini-2.5-flash",
                 temperature: float = 0.1):
        """
        Initialize embedding and LLM models.
        Uses local HuggingFace embeddings (no API calls, no rate limits).
        
        Args:
            embedding_model: HuggingFace embedding model to use
            llm_model: Gemini LLM model to use for generation
            temperature: LLM temperature (lower = more deterministic)
        """
        # Get API key for LLM only
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Initialize LOCAL embedding model (runs on your machine)
        # BAAI/bge-small-en-v1.5 is small, fast, and effective
        print("Loading local embedding model (this may take a moment on first run)...")
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model
        )
        
        # Initialize LLM (still uses API for generation only)
        self.llm = Gemini(
            model=llm_model,
            api_key=api_key,
            temperature=temperature
        )
        
        # Set global settings for LlamaIndex
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        
        print(f"Initialized local embeddings: {embedding_model}")
        print(f"Initialized LLM: {llm_model}")
    
    def get_embedding_model(self):
        """Return the embedding model instance"""
        return self.embed_model
    
    def get_llm(self):
        """Return the LLM instance"""
        return self.llm