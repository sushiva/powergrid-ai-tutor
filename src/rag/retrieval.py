"""
Retrieval module for finding relevant document chunks.
"""

from typing import List, Optional, Dict
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter, FilterOperator


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
    
    def retrieve(self, query: str, filters: Optional[Dict[str, str]] = None) -> List[NodeWithScore]:
        """
        Retrieve relevant chunks for a query with optional metadata filtering.

        Process:
        1. Retrieve chunks using similarity search
        2. Apply metadata filters (if provided) by filtering results
        3. Return top_k most similar filtered chunks

        Args:
            query: User's question
            filters: Optional dictionary of metadata filters
                    e.g., {"topic": "solar", "source": "paper_name.pdf"}

        Returns:
            List of NodeWithScore objects (chunks with similarity scores)
        """
        # If filters provided, retrieve more chunks then filter
        if filters:
            # Retrieve more chunks to ensure we have enough after filtering
            # Get 3x the requested amount to account for filtering
            large_retriever = self.index.as_retriever(similarity_top_k=self.top_k * 3)
            all_nodes = large_retriever.retrieve(query)

            # Filter nodes by metadata
            filtered_nodes = self._filter_nodes_by_metadata(all_nodes, filters)

            # Return top_k from filtered results
            nodes = filtered_nodes[:self.top_k]
        else:
            # Retrieve without filters
            nodes = self.retriever.retrieve(query)

        return nodes

    def _filter_nodes_by_metadata(self, nodes: List[NodeWithScore], filters: Dict[str, str]) -> List[NodeWithScore]:
        """
        Filter nodes based on metadata criteria.

        Args:
            nodes: List of nodes to filter
            filters: Dictionary of metadata key-value pairs to match

        Returns:
            Filtered list of nodes matching all filter criteria
        """
        filtered_nodes = []
        for node in nodes:
            # Check if node matches all filter criteria
            matches_all = True
            for key, value in filters.items():
                if key not in node.node.metadata or node.node.metadata[key] != value:
                    matches_all = False
                    break

            if matches_all:
                filtered_nodes.append(node)

        return filtered_nodes

    def _build_metadata_filters(self, filters: Dict[str, str]) -> MetadataFilters:
        """
        Build MetadataFilters object from filter dictionary.
        NOTE: Not used with FAISS as it doesn't support metadata filtering.
        Kept for compatibility with other vector stores.

        Args:
            filters: Dictionary of field:value pairs to filter on

        Returns:
            MetadataFilters object for LlamaIndex retrieval
        """
        filter_list = []
        for key, value in filters.items():
            if value:  # Only add non-empty values
                filter_list.append(
                    ExactMatchFilter(key=key, value=value)
                )

        return MetadataFilters(filters=filter_list)
    
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