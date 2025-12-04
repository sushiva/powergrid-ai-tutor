"""
Complete RAG pipeline that orchestrates all components.
"""

from typing import Optional
from llama_index.core import VectorStoreIndex
from src.data.loaders import DocumentLoader
from src.data.chunkers import TextChunker
from src.data.embedders import EmbeddingManager
from src.vector_store.faiss_store import FAISSVectorStore
from src.rag.retrieval import Retriever
from src.rag.generator import Generator


class RAGPipeline:
    """
    End-to-end RAG pipeline for document question answering.
    """
    
    def __init__(self):
        """Initialize the RAG pipeline components."""
        self.vector_store = None
        self.index = None
        self.retriever = None
        self.generator = None
        self.embed_manager = None
    
    def build_from_documents(self, pdf_path: str, chunk_size: int = 512):
        """
        Build the complete pipeline from a PDF document.
        
        Args:
            pdf_path: Path to PDF file
            chunk_size: Size of text chunks
        """
        print("Building RAG pipeline from documents...")
        print("=" * 50)
        
        # Step 1: Load documents
        print("\n1. Loading documents...")
        loader = DocumentLoader()
        documents = loader.load_pdf(pdf_path)
        print(f"   Loaded {len(documents)} pages")
        
        # Step 2: Chunk documents
        print("\n2. Chunking documents...")
        chunker = TextChunker(chunk_size=chunk_size)
        nodes = chunker.chunk_documents(documents)
        stats = chunker.get_chunk_info(nodes)
        print(f"   Created {stats['total_chunks']} chunks")
        
        # Step 3: Initialize embeddings
        print("\n3. Initializing embedding model...")
        self.embed_manager = EmbeddingManager()
        
        # Step 4: Build vector store
        print("\n4. Building vector store...")
        self.vector_store = FAISSVectorStore()
        self.index = self.vector_store.build_index(nodes)
        
        # Step 5: Save vector store
        print("\n5. Saving vector store...")
        self.vector_store.save()
        
        # Step 6: Initialize retriever and generator
        print("\n6. Initializing retriever and generator...")
        self.retriever = Retriever(self.index, top_k=5)
        self.generator = Generator(self.index)
        
        print("\n" + "=" * 50)
        print("RAG pipeline ready!")
        print("=" * 50)
    
    def load_existing(self, persist_dir: str = "data/vector_stores/faiss"):
        """
        Load an existing vector store instead of building from scratch.
        
        Args:
            persist_dir: Directory where vector store is saved
        """
        print("Loading existing RAG pipeline...")
        print("=" * 50)
        
        # Initialize embeddings (needed for querying)
        print("\n1. Initializing embedding model...")
        self.embed_manager = EmbeddingManager()
        
        # Load vector store
        print("\n2. Loading vector store...")
        self.vector_store = FAISSVectorStore(persist_dir=persist_dir)
        self.index = self.vector_store.load()
        
        # Initialize retriever and generator
        print("\n3. Initializing retriever and generator...")
        self.retriever = Retriever(self.index, top_k=5)
        self.generator = Generator(self.index)
        
        print("\n" + "=" * 50)
        print("RAG pipeline ready!")
        print("=" * 50)
    
    def query(self, question: str, return_sources: bool = False):
        """
        Query the RAG system.
        
        Args:
            question: User's question
            return_sources: Whether to return source chunks
            
        Returns:
            Answer string or dict with answer and sources
        """
        if self.generator is None:
            raise ValueError("Pipeline not initialized. Call build_from_documents() or load_existing() first.")
        
        if return_sources:
            return self.generator.generate_with_sources(question)
        else:
            return self.generator.generate(question)
    
    def retrieve_only(self, question: str):
        """
        Only retrieve relevant chunks without generation.
        
        Args:
            question: User's question
            
        Returns:
            Retrieved nodes with scores
        """
        if self.retriever is None:
            raise ValueError("Pipeline not initialized. Call build_from_documents() or load_existing() first.")
        
        return self.retriever.retrieve(question) 
    
    
    """
    Explanation:
    
    What this does:
    * build_from_documents(): Complete pipeline from PDF to ready system
    * load_existing(): Load saved vector store (faster, no rebuilding)
    * query(): Ask questions and get answers
    * retrieve_only(): Just see what chunks are retrieved

    This orchestrates all components:
    * Loader → Chunker → Embedder → Vector Store → Retriever → Generator
    """