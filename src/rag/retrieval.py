"""
Retrieval module for finding relevant document chunks.
"""

from typing import List
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore


class Retriever:
    """
    Handles retrieval of relevant document chunks from vector store.
    """
    
    def __init__(self, index: VectorStoreIndex, top_k: int = 5):
        """
        Initialize retriever with a vector index.
        
        Args:
            index: VectorStoreIndex to query
            top_k: Number of most relevant chunks to retrieve
        """
        self.index = index
        self.top_k = top_k
        
        # Create retriever from index
        self.retriever = index.as_retriever(similarity_top_k=top_k)
    
    def retrieve(self, query: str) -> List[NodeWithScore]:
        """
        Retrieve relevant chunks for a query.
        
        Process:
        1. Convert query to embedding
        2. Calculate similarity with all chunks
        3. Return top_k most similar chunks
        
        Args:
            query: User's question
            
        Returns:
            List of NodeWithScore objects (chunks with similarity scores)
        """
        # Retrieve relevant nodes
        # This automatically:
        # - Embeds the query
        # - Searches vector store
        # - Returns top_k results with scores
        nodes = self.retriever.retrieve(query)
        
        return nodes
    
    def get_retrieved_text(self, nodes: List[NodeWithScore]) -> str:
        """
        Extract text content from retrieved nodes.
        
        Args:
            nodes: List of retrieved nodes with scores
            
        Returns:
            Combined text from all retrieved chunks
        """
        texts = []
        for i, node in enumerate(nodes):
            texts.append(f"Chunk {i+1} (Score: {node.score:.4f}):\n{node.text}\n")
        
        return "\n---\n".join(texts)
    
    """
    Explanation:
    
    What this does:
    * as_retriever(): Creates a retriever from the vector index
    * retrieve(): Converts query to embedding, finds similar chunks
    * similarity_top_k: Returns the top K most similar chunks
    * get_retrieved_text(): Helper to see what was retrieved
    
    The retrieval process:
    * Query: "What is solar energy?"
    * Embedding: Convert query to 384-dimensional vector
    * Similarity: Calculate distance to all 21 chunk vectors
    * Top K: Return 5 closest chunks
    
    """