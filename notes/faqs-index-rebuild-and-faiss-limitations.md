# FAQs: Index Rebuild and FAISS Limitations

## 1. When to Rebuild an Index

### When Index Rebuild is Required

An index rebuild is necessary when the **source of truth** (documents, chunking logic, or metadata) changes:

1. **Chunking Logic Changes**
   - Modify chunk size (e.g., 512 to 1024 tokens)
   - Change chunk overlap (e.g., 50 to 100 tokens)
   - Switch splitting strategy (sentence-based to semantic)

2. **Metadata Changes** (Why we rebuilt for Phase 4a)
   - Add new metadata fields (topic, source, author, etc.)
   - Modify metadata extraction logic
   - Change metadata format or structure

3. **New Documents Added**
   - Add new PDFs to the corpus
   - Remove outdated documents
   - Update existing documents

4. **Embedding Model Changes**
   - Switch from one model to another
   - Change embedding dimensions (384 to 768)
   - Upgrade model version

5. **Index Corruption or Migration**
   - Index file corruption
   - Switch vector store (FAISS to Pinecone)
   - Change index type (IndexFlatL2 to IndexIVFFlat)

### Why We Rebuilt for Phase 4a

**The Problem:**
- Modified `src/data/chunkers.py` to add metadata extraction
- Added `_infer_topic()` method for keyword-based topic classification
- Added metadata enrichment in `chunk_documents()` method

**Old Index (Before Phase 4a):**
```python
# Chunks had minimal metadata
{
    'file_name': 'solar_mppt_paper.pdf',
    'page_label': '1'
    # NO 'topic' field
    # NO 'source' field
}
```

**New Code Added:**
```python
# src/data/chunkers.py lines 57-67
for node in nodes:
    source_file = node.metadata.get('file_name', 'unknown')
    topic = self._infer_topic(source_file, node.text)  # NEW

    node.metadata['source'] = source_file  # NEW
    node.metadata['topic'] = topic  # NEW
```

**New Index (After Rebuild):**
```python
# Every chunk now has enriched metadata
{
    'file_name': 'solar_mppt_paper.pdf',
    'page_label': '1',
    'source': 'solar_mppt_paper.pdf',  # NEW
    'topic': 'solar'  # NEW
}
```

**The Consequence:**
- Old index had NO metadata to filter on
- Metadata filtering would return empty results
- Had to rebuild to populate metadata fields

**Rebuild Process:**
```bash
python scripts/data_processing/build_full_index.py
```

**Result:**
- 852 PDF pages loaded
- 2,166 chunks created with metadata
- Each chunk now has `topic` and `source` fields
- Index saved to `data/vector_stores/faiss_full/`
- Took approximately 11 minutes

### When Rebuild is NOT Required

You do NOT need to rebuild if:
- UI changes only (Gradio interface modifications)
- Retrieval logic changes (top_k parameter, reranking)
- Generation prompt changes (LLM instructions)
- Application logic changes (filters, API endpoints)

These changes affect **how you use** the index, not the index itself.

---

## 2. FAISS Limitations and Our Workaround

### What is FAISS?

FAISS (Facebook AI Similarity Search) is a library for efficient vector similarity search:
- Optimized for dense vector search
- Very fast on CPU and GPU
- Used by production systems at scale
- Open source, no cost

### FAISS Limitations

#### Limitation 1: No Native Metadata Filtering

FAISS only stores and searches vectors, not metadata:

```python
# What FAISS stores:
vector_id: [0.23, 0.45, 0.12, ...]  # 384-dimensional vector

# What FAISS does NOT store:
metadata: {'topic': 'solar', 'source': 'paper.pdf'}  # Not in FAISS
```

**Why?** FAISS is designed for pure vector similarity, not structured queries.

#### Limitation 2: IndexFlatL2 is Simple

We use `IndexFlatL2` (exact L2 distance search):
- Searches ALL vectors every time
- No support for pre-filtering
- No query optimization for metadata

Advanced FAISS indexes (IVF, HNSW) also don't support metadata filtering.

#### Limitation 3: Post-Processing Required

To filter by metadata, you must:
1. Retrieve vectors from FAISS
2. Look up metadata separately (stored by LlamaIndex)
3. Filter results in Python
4. Return filtered subset

### What We Tried (That Failed)

**Attempt 1: Pass Filters to LlamaIndex Retriever**
```python
# src/rag/retrieval.py
filters = MetadataFilters(filters=[ExactMatchFilter(key="topic", value="solar")])
retriever = index.as_retriever(filters=filters)

# Error: ValueError: Metadata filters not implemented for Faiss yet.
```

**Why it failed:** LlamaIndex's FAISS integration doesn't support metadata filtering.

**Attempt 2: Use FAISS Pre-Filtering**
```python
# Try to filter before similarity search
faiss_index.search_with_metadata(query_vector, metadata={'topic': 'solar'})

# Not supported - FAISS has no metadata concept
```

**Why it failed:** FAISS API has no metadata filtering capability.

### Our Workaround: Post-Retrieval Filtering

**Implementation:** `src/rag/retrieval.py:29-62`

**Strategy:**
1. Retrieve MORE chunks than needed (3x buffer)
2. Filter by metadata AFTER retrieval (in Python)
3. Return top-k from filtered results

**Code Flow:**
```python
def retrieve(self, query: str, filters: Optional[Dict[str, str]] = None):
    if filters:
        # Step 1: Retrieve 3x chunks (15 instead of 5)
        large_retriever = self.index.as_retriever(similarity_top_k=self.top_k * 3)
        all_nodes = large_retriever.retrieve(query)

        # Step 2: Filter by metadata in Python
        filtered_nodes = self._filter_nodes_by_metadata(all_nodes, filters)

        # Step 3: Return top-k from filtered set
        nodes = filtered_nodes[:self.top_k]
    else:
        # No filters: standard retrieval
        nodes = self.retriever.retrieve(query)

    return nodes
```

**Filtering Logic:**
```python
def _filter_nodes_by_metadata(self, nodes, filters):
    filtered_nodes = []
    for node in nodes:
        # Check if node matches ALL filter criteria
        matches_all = True
        for key, value in filters.items():
            if key not in node.node.metadata or node.node.metadata[key] != value:
                matches_all = False
                break

        if matches_all:
            filtered_nodes.append(node)

    return filtered_nodes
```

### Example: How It Works

**Query:** "What are MPPT algorithms?"
**Filter:** `{"topic": "solar"}`

**Step-by-Step:**

1. **FAISS Semantic Search** (retrieves 15 chunks):
   ```
   solar_chunk_1      (score: 0.92)
   wind_chunk_1       (score: 0.89)
   solar_chunk_2      (score: 0.87)
   battery_chunk_1    (score: 0.85)
   solar_chunk_3      (score: 0.83)
   grid_chunk_1       (score: 0.81)
   solar_chunk_4      (score: 0.78)
   wind_chunk_2       (score: 0.76)
   solar_chunk_5      (score: 0.75)
   battery_chunk_2    (score: 0.73)
   ...
   ```

2. **Python Metadata Filtering** (filter to `topic='solar'`):
   ```
   solar_chunk_1      (score: 0.92)  ✓
   solar_chunk_2      (score: 0.87)  ✓
   solar_chunk_3      (score: 0.83)  ✓
   solar_chunk_4      (score: 0.78)  ✓
   solar_chunk_5      (score: 0.75)  ✓
   ```

3. **Return Top-5 Solar Chunks**

### Trade-offs Analysis

| Aspect | Native Filtering (Ideal) | Post-Retrieval (Our Approach) |
|--------|---------------------------|-------------------------------|
| **Performance** | Faster (filter during search) | Slightly slower (retrieve 3x) |
| **Accuracy** | Perfect (search filtered space) | Good (might miss if <5 in top-15) |
| **Complexity** | Requires special vector DB | Simple, works with FAISS |
| **Cost** | May require paid service | Free (FAISS is open source) |
| **Implementation** | Need migration (Pinecone/Weaviate) | Works with existing setup |
| **Scalability** | Better for large datasets | Good for our size (2,166 chunks) |

### Why This Works Well for Us

1. **Small Dataset:** 2,166 chunks - retrieving 15 vs 5 is negligible
2. **Well-Distributed Topics:**
   - General: 26.6% (577 chunks)
   - Solar: 22.4% (486 chunks)
   - Grid: 18.2% (395 chunks)
   - Battery: 16.7% (361 chunks)
   - Wind: 16.0% (347 chunks)

3. **High Success Rate:** 3x buffer (15 chunks) usually contains 5+ matching chunks
4. **No Migration Needed:** Keeps existing FAISS infrastructure
5. **Simple Implementation:** ~30 lines of Python code

### Alternative Solutions (Why We Rejected Them)

#### Option 1: Switch to Pinecone/Weaviate/Qdrant
**Pros:**
- Native metadata filtering
- Better scalability
- Production-ready

**Cons:**
- Requires complete migration
- Monthly cost ($70-200/month)
- Overkill for 2,166 chunks
- Added complexity (API keys, cloud dependency)

**Verdict:** Rejected - too complex for our use case

#### Option 2: Separate Metadata Database
**Approach:**
- Store vectors in FAISS
- Store metadata in PostgreSQL/SQLite
- Query metadata first, then FAISS

**Cons:**
- Two systems to maintain
- Complex synchronization
- Slower (two queries instead of one)

**Verdict:** Rejected - over-engineered

#### Option 3: Pre-Filter Before Retrieval
**Approach:**
- Build separate FAISS index per topic
- Load topic-specific index based on filter

**Cons:**
- 5x storage (one index per topic)
- Complex index management
- Slow index switching
- Can't combine filters (topic + source)

**Verdict:** Rejected - inflexible and wasteful

### Performance Impact

**Baseline (No Filters):**
- Retrieve 5 chunks
- Latency: ~50ms

**With Filters (Post-Retrieval):**
- Retrieve 15 chunks (FAISS)
- Filter 15 chunks (Python)
- Return 5 chunks
- Latency: ~65ms (+30% overhead)

**Conclusion:** 15ms overhead is negligible for our use case.

### When to Consider Migration

Consider switching to a native metadata-filtering vector store if:
- Dataset grows beyond 100,000 chunks
- Need complex filters (AND/OR/NOT, ranges, fuzzy matching)
- Latency becomes critical (<10ms required)
- Budget allows for paid service
- Multiple metadata dimensions (topic + author + date + category)

For our current project (2,166 chunks, simple filters), post-retrieval filtering is the right choice.

---

## Summary

### Index Rebuild
- **When:** Chunking logic, metadata, documents, or embeddings change
- **Why for Phase 4a:** Added `topic` and `source` metadata to chunks
- **Process:** Run `build_full_index.py` to regenerate all embeddings
- **Result:** All 2,166 chunks now have filterable metadata

### FAISS Limitations
- **Problem:** FAISS doesn't support native metadata filtering
- **Solution:** Post-retrieval filtering (retrieve 3x, filter, return top-k)
- **Trade-off:** Slight performance overhead for simplicity and cost savings
- **Why it works:** Small dataset, well-distributed topics, simple implementation

Both decisions prioritize **simplicity, maintainability, and cost-effectiveness** over perfect optimization.
