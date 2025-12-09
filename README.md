# ⚡ PowerGrid AI Tutor

An advanced Retrieval-Augmented Generation (RAG) system for electrical engineering and renewable energy education.

**Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/sudhirshivaram/powergrid-ai-tutor)  
**Author:** Sudhir Shivaram  
**Built for:** LLM Developer Certification - Advanced RAG Project

---

## Features

- 🔋 **50+ Research Papers** on solar, wind, battery, and grid technologies
- 🔍 **Advanced RAG Pipeline**: Query expansion, hybrid search (BM25 + semantic), LLM reranking
- 📊 **Evaluated System**: 70% hit rate, 55% MRR on test queries
- 🎯 **Domain-Specific**: Electrical engineering, renewable energy, power systems, smart grids
- ⚙️ **Configurable**: Toggle features via UI, runtime API key input
- 💰 **Cost-Optimized**: ~$0.001-0.002 per query (well under $0.50 requirement)

---

## Quick Start

### Using the HuggingFace Space (Recommended)

1. **Visit**: https://huggingface.co/spaces/sudhirshivaram/powergrid-ai-tutor
2. **Choose LLM Provider**:
   - **OpenAI** (recommended): Get API key at [OpenAI Platform](https://platform.openai.com/api-keys)
   - **Gemini**: Get free key at [Google AI Studio](https://aistudio.google.com/app/apikey)
3. **Enter API Key** and click "Initialize System"
4. **Select Features**: Toggle query expansion, hybrid search, reranking
5. **Ask Questions** about renewable energy, power systems, or smart grids!

### Local Setup

```bash
# Clone repository
git clone https://github.com/sushiva/powergrid-ai-tutor.git
cd powergrid-ai-tutor

# Install dependencies
pip install -r requirements.txt

# Set API key (choose one)
export OPENAI_API_KEY="your-key-here"
# OR
export GOOGLE_API_KEY="your-key-here"

# Run locally
python app.py
```

---

## Example Questions

**Renewable Energy:**
- "How do wind turbines generate electricity?"
- "What are the latest advances in battery storage for renewable energy?"
- "Explain the challenges of integrating solar power into the grid"

**Power Systems:**
- "How does a transformer reduce voltage in distribution systems?"
- "What is demand response in smart grids?"
- "Explain three-phase power transmission"

**Smart Grids:**
- "What are the benefits of smart grid technology?"
- "How do microgrids improve grid resilience?"
- "Explain vehicle-to-grid (V2G) systems"

---

## System Architecture

### Knowledge Base
- **50 ArXiv Papers** (852 pages, 2,166 chunks)
- **Topics**: Solar PV, wind energy, battery storage, grid integration, smart grids
- **Chunking**: 512 tokens with 50-token overlap
- **Metadata**: Topic tags, source types, authors

### RAG Pipeline

1. **Query Processing**
   - Query expansion (multi-query generation)
   - Intent classification (optional router)

2. **Retrieval**
   - **Semantic Search**: BAAI/bge-small-en-v1.5 embeddings + FAISS
   - **Keyword Search**: BM25 for exact term matching
   - **Hybrid Fusion**: Combines both with weighted scoring

3. **Reranking**
   - Cohere reranker for relevance refinement
   - Top-3 sources selected

4. **Generation**
   - **LLM**: OpenAI GPT-4o-mini or Google Gemini 1.5 Flash
   - **Prompt Engineering**: Context-only answers, hallucination prevention
   - **Source Attribution**: Shows document sources with relevance scores

### Cost Breakdown
- **Embedding**: BAAI/bge-small-en-v1.5 (local, $0)
- **Reranking**: Cohere API (~$0.0001 per query)
- **LLM**: GPT-4o-mini (~$0.0015 per query) or Gemini (~$0.0003 per query)
- **Total**: ~$0.001-0.002 per query

---

## Advanced Features (8 Implemented)

1. **Query Expansion** - Generates multiple query variations for better retrieval
2. **Hybrid Search** - Combines semantic (vector) + keyword (BM25) search
3. **Reranking** - Uses Cohere to refine relevance of retrieved chunks
4. **Metadata Filtering** - Filter by topic (Solar/Wind/Battery/Grid) or source type
5. **Source Attribution** - Shows document sources with relevance percentages
6. **Chat History** - Maintains conversation context across turns
7. **Multi-Provider LLM** - Runtime selection between OpenAI and Gemini
8. **Configurable Pipeline** - Toggle features ON/OFF via UI

---

## Evaluation Results

**Test Set**: 20 domain-specific questions

| Metric | Score |
|--------|-------|
| Hit Rate@3 | 70% |
| MRR@3 | 55% |
| Avg Response Time | 2-4s |
| Cost per Query | $0.001-0.002 |

**Key Findings:**
- Hybrid search improves recall by 15% over semantic-only
- Reranking improves precision by 20%
- Query expansion helps with ambiguous queries
- Prompt engineering critical for preventing hallucinations

---

## Prompt Engineering Journey

### The Pasta Hallucination Problem
Initial testing revealed the system answered "How to cook pasta?" with a correct cooking recipe, citing solar panel PDFs with 85-90% relevance scores. This showed that **semantic similarity ≠ domain relevance**.

### Evolution
1. **v1 - Too Strict**: Rejected valid in-domain questions
2. **v2 - Too Lenient**: Answered out-of-domain questions
3. **v3 - Balanced (Final)**: 
   - Uses context even if partial
   - Handles broad questions gracefully
   - Only rejects truly unrelated topics

### Final Prompt Strategy
```
- Answer using provided context, even if partial or general
- For broad questions (e.g., "batteries"), provide available info
- Only reject if completely unrelated (cooking, sports, etc.)
- Hide sources when rejecting questions
```

---

## Project Structure

```
powergrid-ai-tutor/
├── app/
│   ├── main.py              # Gradio UI with API key input
│   └── config.py            # Configuration settings
├── src/
│   ├── data/
│   │   ├── loaders.py       # PDF loading
│   │   ├── chunkers.py      # Text chunking
│   │   └── embedders.py     # Embedding generation
│   ├── rag/
│   │   ├── pipeline.py      # Main RAG orchestration
│   │   ├── retrieval.py     # Hybrid retrieval
│   │   ├── reranker.py      # Cohere reranking
│   │   ├── query_expander.py # Query expansion
│   │   └── generator.py     # LLM answer generation
│   └── vector_store/
│       └── faiss_store.py   # FAISS index management
├── data/
│   ├── raw/papers/          # Source PDFs
│   └── vector_stores/faiss_full/  # Prebuilt FAISS index
├── evaluation/
│   ├── run_evaluation.py    # Evaluation script
│   └── datasets/            # Test queries & ground truth
├── app.py                   # Main entry point
├── requirements.txt         # Dependencies
└── README.md               # This file
```

---

## API Keys Setup

### OpenAI (Recommended)
1. Sign up at https://platform.openai.com
2. Add payment method (pay-as-you-go)
3. Create API key: https://platform.openai.com/api-keys
4. **Cost**: ~$0.0015 per query (GPT-4o-mini)

### Google Gemini
1. Visit https://aistudio.google.com/app/apikey
2. Create free API key (no payment required)
3. **Free Tier**: 15 requests/minute, 1500/day
4. **Cost**: ~$0.0003 per query (Gemini 1.5 Flash)
5. **Note**: May have regional restrictions on HuggingFace servers

---

## Known Limitations

1. **Broad Single-Term Queries**: Questions like "batteries" may return partial information if context is limited
2. **Regional Restrictions**: Gemini may not work on all HuggingFace servers (use OpenAI)
3. **Response Time**: 2-4 seconds per query (trade-off for quality)
4. **Domain Boundaries**: Borderline topics (general electrical engineering) may be rejected

---

## Deployment

### HuggingFace Spaces
- **URL**: https://huggingface.co/spaces/sudhirshivaram/powergrid-ai-tutor
- **Framework**: Gradio 6.0.2
- **Hardware**: CPU Basic (free tier)
- **Features**: Runtime API key input, no hardcoded secrets

### Local Deployment
```bash
# Run with Gradio
python app.py

# Access at http://localhost:7860
```

---

## Testing Checklist

**In-Domain (Should Answer):**
- ✅ "Tell me about batteries"
- ✅ "How does a transformer work?"
- ✅ "Explain smart grid technology"
- ✅ "Solar panel efficiency challenges"

**Out-of-Domain (Should Reject):**
- ✅ "How to cook pasta?"
- ✅ "LED bulbs at home?"
- ✅ "Stadium electricity consumption?"

**Features:**
- ✅ All ON: Better answers, 3-4s response time
- ✅ All OFF: Faster (2-3s), decent quality
- ✅ Processing time displayed
- ✅ Relevance scores shown (70-90% range)

---

## License

MIT License - See [LICENSE](LICENSE) for details

---

## Author

**Sudhir Shivaram**  
**Project**: LLM Developer Certification - Advanced RAG System  
**Date**: December 2025

---

## Acknowledgments

- ArXiv for open research papers
- LlamaIndex for RAG framework
- Cohere for reranking API
- HuggingFace for deployment platform
