"""
Complete RAG pipeline that orchestrates all components.
"""

from typing import Optional, Dict
from llama_index.core import VectorStoreIndex
from src.data.loaders import DocumentLoader
from src.data.chunkers import TextChunker
from src.data.embedders import EmbeddingManager
from src.vector_store.faiss_store import FAISSVectorStore
from src.rag.retrieval import Retriever
from src.rag.generator import Generator
from src.rag.reranker import Reranker


class RAGPipeline:
    """
    End-to-end RAG pipeline for document question answering.
    """
    
    def __init__(self, use_reranking: bool = False):
        """
        Initialize the RAG pipeline components.

        Args:
            use_reranking: Whether to use LLM reranking for better relevance
        """
        self.vector_store = None
        self.index = None
        self.retriever = None
        self.generator = None
        self.embed_manager = None
        self.reranker = None
        self.use_reranking = use_reranking
    
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
        # If using reranking, retrieve more chunks initially (top-10)
        # Then rerank to get best 5
        initial_top_k = 10 if self.use_reranking else 5
        self.retriever = Retriever(self.index, top_k=initial_top_k)
        self.generator = Generator(self.index)

        # Initialize reranker if enabled
        if self.use_reranking:
            self.reranker = Reranker(top_n=5)

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
        # If using reranking, retrieve more chunks initially (top-10)
        # Then rerank to get best 5
        initial_top_k = 10 if self.use_reranking else 5
        self.retriever = Retriever(self.index, top_k=initial_top_k)
        self.generator = Generator(self.index)

        # Initialize reranker if enabled
        if self.use_reranking:
            self.reranker = Reranker(top_n=5)

        print("\n" + "=" * 50)
        print("RAG pipeline ready!")
        print("=" * 50)
    
    def query(self, question: str, filters: Optional[Dict[str, str]] = None, return_sources: bool = False):
        """
        Query the RAG system with optional metadata filtering.

        Args:
            question: User's question
            filters: Optional metadata filters (e.g., {"topic": "solar", "source": "paper.pdf"})
            return_sources: Whether to return source chunks

        Returns:
            Answer string or dict with answer and sources
        """
        if self.generator is None:
            raise ValueError("Pipeline not initialized. Call build_from_documents() or load_existing() first.")

        # For filtered queries, we need to use retrieve + generate separately
        # because the generator doesn't directly support filters
        if filters:
            # Retrieve filtered nodes
            nodes = self.retrieve_only(question, filters=filters)

            # Generate answer from filtered nodes
            if return_sources:
                # Build context from nodes
                context = "\n\n".join([node.node.text for node in nodes])
                answer = self.generator.llm.complete(
                    f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based only on the context above:"
                ).text
                return {"answer": answer, "sources": nodes}
            else:
                # Build context from nodes
                context = "\n\n".join([node.node.text for node in nodes])
                answer = self.generator.llm.complete(
                    f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based only on the context above:"
                ).text
                return answer
        else:
            # Use standard generation without filters
            if return_sources:
                return self.generator.generate_with_sources(question)
            else:
                return self.generator.generate(question)
    
    def retrieve_only(self, question: str, filters: Optional[Dict[str, str]] = None):
        """
        Only retrieve relevant chunks without generation.

        Args:
            question: User's question
            filters: Optional metadata filters

        Returns:
            Retrieved nodes with scores (reranked if enabled, filtered by metadata)
        """
        if self.retriever is None:
            raise ValueError("Pipeline not initialized. Call build_from_documents() or load_existing() first.")

        # Retrieve initial chunks with optional filters
        nodes = self.retriever.retrieve(question, filters=filters)

        # Apply reranking if enabled
        if self.use_reranking and self.reranker:
            nodes = self.reranker.rerank(nodes, question)

        return nodes 
    
    
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