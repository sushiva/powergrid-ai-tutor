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

## 3. Hybrid Search Performance (Phase 4b)

### Question: Does Hybrid Search Make Queries Slower?

**Short Answer:** No - it's actually slightly **faster** on average!

### Benchmark Results (Actual Measurements)

Tested with 5 different queries on 2,166 document chunks:

| Query Type | Semantic Only | Hybrid Search | Difference |
|------------|---------------|---------------|------------|
| "What are MPPT algorithms?" | 44.9ms | 44.3ms | -0.6ms (-1.4%) |
| "battery energy storage system" | 30.7ms | 38.3ms | +7.6ms (+24.6%) |
| "wind turbine power generation" | 48.6ms | 52.0ms | +3.4ms (+7.0%) |
| "solar panel efficiency factors" | 72.6ms | 56.6ms | -16.0ms (-22.0%) |
| "grid stability regulation" | 63.3ms | 55.8ms | -7.5ms (-11.9%) |
| **Average** | **52.0ms** | **49.4ms** | **-2.6ms (-5% faster!)** |

### Key Findings

1. **Hybrid is 5% faster on average**
   - Expected it to be slower (semantic + BM25 + RRF)
   - BM25 keyword search is extremely fast
   - RRF fusion overhead is negligible

2. **Variability is normal**
   - Some queries faster, some slower
   - Difference ranges from -16ms to +7.6ms
   - System factors (disk I/O, caching) dominate

3. **Both are fast enough**
   - All queries complete in <100ms
   - Real-time performance for both methods

### Initialization Overhead

- **BM25 index building:** +0.4 seconds (one-time on pipeline load)
- **Memory:** <10MB additional
- **Trade-off:** Negligible cost for better accuracy

### Why Hybrid Can Be Faster

This counterintuitive result happens because:

1. **BM25 is extremely fast**
   - Simple term matching in pre-tokenized corpus
   - No complex vector operations
   - Pure Python dictionary lookups

2. **RRF fusion is trivial**
   - Just scoring and sorting
   - Typically <5ms overhead

3. **Parallel execution benefits**
   - Semantic and BM25 searches may benefit from CPU parallelization
   - Modern CPUs handle both operations efficiently

4. **System-level factors**
   - Disk I/O caching
   - CPU cache hits
   - Memory access patterns
   - These factors create more variability than the algorithm choice

### Recommendation

**Enable hybrid search by default:**
- ✓ No performance penalty (actually faster!)
- ✓ Better accuracy for keyword-heavy queries
- ✓ Handles acronyms and technical terms better
- ✓ Minimal initialization cost (+0.4s)
- ✓ Negligible memory footprint (<10MB)

**Only use semantic-only if:**
- Extremely resource-constrained environment
- Very small corpus (<100 documents)
- The 0.4s initialization time matters

### Implementation

```python
# Enable hybrid search in pipeline
pipeline = RAGPipeline(use_hybrid=True)
pipeline.load_existing()

# Queries automatically use BM25 + Semantic + RRF
answer = pipeline.query("What are MPPT algorithms?")
```

### Related Documentation

See [notes/phase4b-hybrid-search.md](phase4b-hybrid-search.md) for complete implementation details.

---

## 4. Ollama and Local LLMs (Phase 6)

### Question: What is the difference between Ollama and Qwen?

**Short Answer:** Ollama is a platform, Qwen is a model.

**Detailed Explanation:**

Think of it like this:
- **Ollama** = Spotify (the platform/app)
- **Qwen 2.5** = A specific song you play on Spotify

**Ollama** is a platform (like Docker) that lets you:
- Download and run any open-source LLM locally
- Manage different models
- Provide a consistent API for all models
- Handle the infrastructure (GPU/CPU management, memory)

**Qwen 2.5** is an actual AI model created by Alibaba Cloud:
- One of many models you can run on Ollama
- Comes in different sizes (1.5B, 7B, 14B parameters)
- Excellent for technical content and reasoning

**Other Models You Can Run on Ollama:**
- Llama 3.1 (Meta) - general purpose
- Mistral (Mistral AI) - fast and efficient
- Gemma (Google) - lightweight
- Phi-3 (Microsoft) - small but capable
- 100+ more models

**Analogy:**
```
Ollama Platform
├── Qwen 2.5 7B (what we're using)
├── Llama 3.1 8B
├── Mistral 7B
├── Gemma 2B
└── ... many more
```

### Question: How much does Ollama cost compared to Gemini?

**Cost Comparison:**

**Gemini API (API-based):**
- **Setup Cost:** $0 (just need API key)
- **Per Query Cost:** ~$0.003 per query
- **Monthly Estimates:**
  - 100 queries: $0.30
  - 1,000 queries: $3.00
  - 10,000 queries: $30.00
  - 100,000 queries: $300.00

**Pricing Breakdown (Gemini 2.5 Flash):**
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens
- Average RAG query: ~1,500 input + ~500 output tokens = $0.003

**Ollama (Local):**
- **Setup Cost:** $0 (free download, 4.7GB model)
- **Per Query Cost:** $0 (effectively free)
- **Ongoing Costs:** ~$0.001 per query in electricity (negligible)

**Break-Even Analysis:**
- Ollama setup time: ~30 minutes
- Gemini cost: $3.00 per 1,000 queries
- **Break-even: ~1,000 queries** (if your time is worth $100/hour)
- After 1,000 queries, Ollama saves money

**When Each is Worth It:**

**Use Gemini when:**
- Low query volume (<1,000/month) - costs <$3
- Need fastest responses (2-3s vs 30-40s)
- Limited local hardware (low RAM, no GPU)
- Production app with SLA requirements

**Use Ollama when:**
- High query volume (>1,000/month) - saves significant money
- Cost-sensitive project or free tier requirement
- Privacy critical (data never leaves machine)
- Unlimited testing/development needed
- Have decent hardware (8GB+ RAM, ideally GPU)

**Real-World Example:**
```
Scenario: Educational app with 10,000 student queries/month

Gemini Cost: 10,000 × $0.003 = $30/month = $360/year
Ollama Cost: $0/month (just electricity ~$1/month) = ~$12/year

Annual Savings: $348/year with Ollama
```

**Performance Comparison (Actual Benchmark):**

| Metric | Gemini (API) | Ollama (Local) | Winner |
|--------|--------------|----------------|--------|
| **Response Time** | 2.15s | 38.56s | Gemini (18x faster) |
| **Answer Quality** | Excellent | Very Good | Gemini (slightly better) |
| **Cost per 1000 queries** | $3.00 | $0.00 | Ollama (free) |
| **Rate Limits** | 15-20/min | Unlimited | Ollama |
| **Privacy** | Data sent to Google | 100% private | Ollama |
| **Setup Complexity** | Easy (API key) | Medium (install + download) | Gemini |
| **Hardware Requirements** | None | 8GB+ RAM recommended | Gemini |

### Question: How did we overcome Gemini rate limit challenges?

**The Rate Limit Problem:**

Gemini Free Tier has strict rate limits:
- **15-20 requests per minute**
- **1,500 requests per day**

**Why We Hit Rate Limits Initially:**

Our full RAG pipeline (with all features enabled) makes **3 LLM calls per query**:

```python
pipeline = RAGPipeline(
    use_query_expansion=True,    # LLM Call #1
    use_hybrid=True,              # No LLM call (just BM25 + semantic)
    use_reranking=True            # LLM Call #2
)
answer = pipeline.query("What are MPPT algorithms?")  # LLM Call #3 (generation)
```

**The 3 LLM Calls:**
1. **Query Expansion** (Phase 5a): LLM generates technical terms/synonyms (~50 tokens)
   - Example: "solar panels" → adds "PV", "photovoltaic", "solar cells"
   - File: `src/rag/query_expander.py`

2. **Reranking** (Phase 4c): LLM scores retrieved chunks for relevance (~500-1000 tokens)
   - Takes top 10 chunks, returns top 5 most relevant
   - File: `src/rag/reranker.py`

3. **Answer Generation**: LLM synthesizes final answer from chunks (~1000-2000 tokens)
   - Built into LlamaIndex query engine

**Different Pipeline Configurations:**

```python
# Minimal (1 LLM call) - Only generation
pipeline = RAGPipeline()

# With Reranking (2 LLM calls) - Reranking + Generation
pipeline = RAGPipeline(use_reranking=True)

# Full Pipeline (3 LLM calls) - Expansion + Reranking + Generation
pipeline = RAGPipeline(
    use_query_expansion=True,
    use_hybrid=True,
    use_reranking=True
)
```

When testing with 4 different pipeline configurations:
- Config 1 (baseline): 3 queries × 1 call = 3 calls
- Config 2 (hybrid): 3 queries × 1 call = 3 calls
- Config 3 (hybrid + rerank): 3 queries × 2 calls = 6 calls
- Config 4 (full pipeline): 3 queries × 3 calls = 9 calls
- **Total: 21+ calls in rapid succession** → hits 15-20/min rate limit quickly!

**Solution 1: Use Ollama for Development (Recommended)**

```python
import os

# Use Ollama for unlimited development testing
# Use Gemini only for production
llm_provider = "gemini" if os.getenv("ENVIRONMENT") == "production" else "ollama"

pipeline = RAGPipeline(
    use_query_expansion=True,
    use_hybrid=True,
    use_reranking=True,
    llm_provider=llm_provider  # Automatic switching
)
```

**Benefits:**
- Unlimited testing in development (no rate limits)
- No API costs during development
- Only use Gemini in production (where quality matters most)

**Solution 2: Implement Rate Limiting in Code**

```python
import time
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, max_calls=15, time_window=60):
        self.max_calls = max_calls
        self.time_window = time_window  # seconds
        self.calls = []

    def wait_if_needed(self):
        now = datetime.now()
        # Remove calls outside time window
        self.calls = [call for call in self.calls
                     if now - call < timedelta(seconds=self.time_window)]

        if len(self.calls) >= self.max_calls:
            # Wait until oldest call expires
            wait_time = (self.calls[0] + timedelta(seconds=self.time_window) - now).total_seconds()
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)

        self.calls.append(now)

# Usage
rate_limiter = RateLimiter(max_calls=15, time_window=60)

for query in queries:
    rate_limiter.wait_if_needed()
    answer = pipeline.query(query)
```

**Solution 3: Batch Processing with Delays**

```python
import time

def process_queries_with_delays(queries, pipeline, delay=4):
    """
    Process queries with delays to avoid rate limits.
    delay=4 seconds allows 15 queries/minute (15 × 4 = 60s)
    """
    results = []
    for i, query in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}...")
        result = pipeline.query(query)
        results.append(result)

        if i < len(queries) - 1:  # Don't delay after last query
            time.sleep(delay)

    return results
```

**Solution 4: Fallback System (Hybrid Approach)**

```python
from google.api_core.exceptions import ResourceExhausted

def query_with_fallback(question: str):
    try:
        # Try Gemini first (fast + quality)
        pipeline_gemini = RAGPipeline(llm_provider="gemini")
        pipeline_gemini.load_existing()
        return pipeline_gemini.query(question)
    except ResourceExhausted:
        # Fall back to Ollama if rate limited
        print("Rate limited on Gemini, switching to Ollama...")
        pipeline_ollama = RAGPipeline(llm_provider="ollama")
        pipeline_ollama.load_existing()
        return pipeline_ollama.query(question)
```

**Solution 5: Upgrade to Paid Tier**

**Gemini Paid Tier Benefits:**
- **1,000+ requests per minute** (vs 15-20 free)
- **50,000+ requests per day** (vs 1,500 free)
- Still very affordable (~$3 per 1,000 queries)

**When to Upgrade:**
- Production application with users
- Can't tolerate Ollama's slower response times
- Need guaranteed API availability

**Why Comparison Test Worked:**

The comparison script only made **2 simple queries** (1 Gemini + 1 Ollama):
- 2 queries × 1 LLM call each = **2 total API calls**
- Well under 15-20/min limit
- No query expansion or reranking overhead

**Best Practice Recommendation:**

**For Development:**
```python
pipeline = RAGPipeline(
    use_query_expansion=True,
    use_hybrid=True,
    use_reranking=True,
    llm_provider="ollama"  # Unlimited testing
)
```

**For Production (Low Traffic):**
```python
pipeline = RAGPipeline(
    use_query_expansion=True,
    use_hybrid=True,
    use_reranking=True,
    llm_provider="gemini"  # Best quality + speed
)
# Add rate limiting or upgrade to paid tier
```

**For Production (High Traffic):**
```python
# Option 1: Use Ollama with GPU (fast + free)
pipeline = RAGPipeline(llm_provider="ollama")

# Option 2: Use Gemini paid tier (fast + reliable)
pipeline = RAGPipeline(llm_provider="gemini")
# Upgrade to paid tier for higher limits
```

### Question: Do I need to rebuild the index when switching between Gemini and Ollama?

**No, you do NOT need to rebuild the index.**

**Why:**
- Both Gemini and Ollama use the **same embedding model** (HuggingFace BAAI/bge-small-en-v1.5)
- Embeddings are generated locally (not by the LLM)
- The vector index only stores embeddings, not LLM-specific data

**What Changes:**
- **Embeddings:** Same (same HuggingFace model)
- **Vector Index:** Same (FAISS index unchanged)
- **LLM:** Different (Gemini API vs Ollama local)

**LLM is Only Used For:**
1. Query expansion (generating related terms)
2. Reranking (scoring retrieved chunks)
3. Answer generation (synthesizing final response)

**Code Example:**
```python
# Load same index with different LLM
pipeline_gemini = RAGPipeline(llm_provider="gemini")
pipeline_gemini.load_existing(persist_dir="data/vector_stores/faiss_full")

pipeline_ollama = RAGPipeline(llm_provider="ollama")
pipeline_ollama.load_existing(persist_dir="data/vector_stores/faiss_full")
# Both use the SAME vector index, different LLMs
```

### Speed Optimization for Ollama

If Ollama's 38-second response time is too slow, you have options:

**Option 1: Use Smaller Models**
```bash
# Qwen 2.5 1.5B (much faster, still good quality)
ollama pull qwen2.5:1.5b

# Phi-3 Mini (Microsoft, very fast)
ollama pull phi3:mini
```

```python
# Use smaller model in code
pipeline = RAGPipeline(
    llm_provider="ollama",
    llm_model="qwen2.5:1.5b"  # Faster responses (~15s)
)
```

**Speed Comparison:**
- qwen2.5:7b - 38s (best quality)
- qwen2.5:1.5b - ~15s (good quality)
- phi3:mini - ~8s (decent quality)

**Option 2: GPU Acceleration**

If you have NVIDIA GPU:
```bash
# Ollama automatically uses GPU if available
nvidia-smi  # Check GPU availability

# Same code, much faster (5-10x speedup)
pipeline = RAGPipeline(llm_provider="ollama")
# Will automatically use GPU if detected
```

**With GPU (NVIDIA RTX 3060):**
- qwen2.5:7b - ~4s (comparable to Gemini!)
- qwen2.5:1.5b - ~1.5s (faster than Gemini)

### Related Documentation

See [notes/phase6-local-vs-api-llms.md](phase6-local-vs-api-llms.md) for complete Ollama implementation details, installation guide, and comprehensive comparison.

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

### Hybrid Search Performance
- **Question:** Does hybrid search slow down queries?
- **Answer:** No - it's actually 5% faster on average (49ms vs 52ms)
- **Recommendation:** Enable by default for better accuracy with no performance cost
- **Implementation:** `RAGPipeline(use_hybrid=True)`

### Ollama and Local LLMs
- **Ollama vs Qwen:** Ollama is a platform (like Docker), Qwen is a model you run on it
- **Cost:** Ollama is free after setup, Gemini costs ~$0.003/query (~$3 per 1,000 queries)
- **Performance:** Gemini is 18x faster (2.15s vs 38.56s), but Ollama is unlimited and private
- **Rate Limits:** Overcome by using Ollama for development, Gemini for production
- **Index Rebuild:** NOT required when switching - both use same embeddings (HuggingFace BAAI/bge-small-en-v1.5)
- **Best Practice:** Use Ollama for development (unlimited testing), Gemini for production (speed + quality)

All decisions prioritize **simplicity, maintainability, and cost-effectiveness** over perfect optimization.
