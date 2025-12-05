# Phase 4: Hybrid Search - Implementation Plan

## Overview

Implement hybrid search that combines semantic (vector) search with keyword (BM25) search for better retrieval performance.

## What is Hybrid Search?

**Current System**: Pure semantic search using FAISS
- Converts query to embeddings
- Finds similar vectors
- Good for conceptual queries
- Struggles with exact terms/acronyms

**Hybrid Search**: Combines two retrieval methods
1. **Semantic Search** (existing): Vector similarity
2. **Keyword Search** (new): BM25 algorithm

**Fusion**: Merge and rank results from both methods

## Why Hybrid Search?

### Strengths of Each Approach

| Semantic Search | Keyword Search (BM25) |
|----------------|---------------------|
| Understands concepts | Matches exact terms |
| Handles synonyms | Great for acronyms |
| Context-aware | Fast and precise |
| Weakness: misses exact terms | Weakness: misses concepts |

### Example

**Query**: "What is MPPT in solar systems?"

- **Semantic alone**: Might miss "MPPT" acronym
- **Keyword alone**: Finds "MPPT" but misses related concepts
- **Hybrid**: Finds both exact term AND related context

## Expected Improvements

**Current Performance** (Semantic only):
- Hit Rate: 50.0%
- MRR: 33.9%

**Target Performance** (Hybrid):
- Hit Rate: 60-70%
- MRR: 40-50%

## Implementation Steps

### Phase 1: Add BM25 Retriever (30 min)
1. Install rank_bm25 library
2. Create `src/rag/bm25_retriever.py`
3. Build BM25 index from document chunks
4. Implement keyword search method

### Phase 2: Implement Fusion (20 min)
1. Create `src/rag/hybrid_retriever.py`
2. Implement Reciprocal Rank Fusion (RRF) algorithm
3. Merge results from semantic + keyword search
4. Configure fusion weights

### Phase 3: Integrate into Pipeline (15 min)
1. Update `src/rag/pipeline.py`
2. Add `use_hybrid_search` parameter
3. Initialize BM25 retriever when enabled
4. Update retrieval flow

### Phase 4: Evaluate & Compare (20 min)
1. Create `evaluation/compare_hybrid.py`
2. Run baseline (semantic only)
3. Run hybrid search
4. Compare metrics

### Phase 5: Update UI & Documentation (15 min)
1. Add `--hybrid` flag to UI
2. Update README
3. Commit changes

## Technical Details

### BM25 Index Structure
```python
# Tokenize all chunks
corpus = [chunk.text.split() for chunk in nodes]

# Build BM25 index
from rank_bm25 import BM25Okapi
bm25 = BM25Okapi(corpus)

# Search
query_tokens = query.split()
scores = bm25.get_scores(query_tokens)
```

### Reciprocal Rank Fusion (RRF)
```python
def rrf_fusion(semantic_results, bm25_results, k=60):
    scores = {}

    # Score semantic results
    for rank, doc_id in enumerate(semantic_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # Score BM25 results
    for rank, doc_id in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # Sort by combined score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### Configuration Options
- **Semantic weight**: 0.0 - 1.0 (default: 0.5)
- **BM25 weight**: 0.0 - 1.0 (default: 0.5)
- **RRF k parameter**: 20-100 (default: 60)

## File Structure

```
src/rag/
├── bm25_retriever.py       # NEW - BM25 keyword search
├── hybrid_retriever.py     # NEW - Fusion logic
├── retrieval.py            # EXISTING - Semantic search
├── pipeline.py             # MODIFY - Add hybrid mode
└── reranker.py             # EXISTING - Optional reranking

evaluation/
├── compare_hybrid.py       # NEW - Hybrid vs baseline
└── compare_reranking.py    # EXISTING

app/
└── main.py                 # MODIFY - Add --hybrid flag
```

## Usage Examples

### Run with Hybrid Search
```bash
# Hybrid search only
python app/main.py --hybrid

# Hybrid + Reranking (best quality, slowest)
python app/main.py --hybrid --rerank

# Semantic only (fastest)
python app/main.py
```

### Evaluation
```bash
# Compare all modes
python evaluation/compare_hybrid.py
```

## Trade-offs

| Mode | Hit Rate | MRR | Speed | Best For |
|------|----------|-----|-------|----------|
| Semantic only | 50% | 34% | Fast | Conceptual queries |
| Hybrid | 65%* | 45%* | Medium | Mixed queries |
| Hybrid + Rerank | 60%* | 50%* | Slow | Best quality |

*Estimated improvements

## Success Criteria

- [ ] BM25 retriever implemented
- [ ] Hybrid fusion working
- [ ] Hit Rate > 60%
- [ ] Evaluation comparison complete
- [ ] UI flag added
- [ ] Documentation updated
- [ ] Code committed to Git

## Dependencies

```txt
rank-bm25  # BM25 algorithm implementation
```

## Timeline

- **Total Time**: ~1.5 - 2 hours
- **Difficulty**: Moderate
- **Prerequisites**: Completed Phase 1-3

## Next Steps After Phase 4

1. **Phase 5**: Query Expansion (improve query understanding)
2. **Phase 6**: Response Quality Metrics (faithfulness, relevancy)
3. **Phase 7**: Deployment to HuggingFace Spaces

## References

- BM25 Algorithm: https://en.wikipedia.org/wiki/Okapi_BM25
- Reciprocal Rank Fusion: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- LlamaIndex Hybrid Search: https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/
