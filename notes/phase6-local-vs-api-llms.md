# Phase 6: Local vs API-Based LLMs - Ollama Integration

## Overview

Implemented support for both **API-based LLMs** (Gemini) and **local LLMs** (Ollama) to give users choice between cloud services and on-premise inference.

This allows users to choose based on their priorities:
- **Gemini**: Best quality, fast, but costs money and has rate limits
- **Ollama**: Free, private, unlimited, but slower and requires local resources

## What is Ollama?

**Ollama** is NOT an LLM - it's a **platform** for running LLMs locally.

Think of it like:
- **Ollama** = Docker (the platform)
- **Qwen 2.5** = Your container (the actual AI model)

**Key Features:**
- Download and run any open-source LLM locally
- No API keys needed
- 100% private - data never leaves your machine
- Unlimited usage - no rate limits
- Free after initial setup

**Supported Models:**
- Qwen 2.5 (Alibaba Cloud) - excellent for technical content
- Llama 3.1 (Meta) - general purpose
- Mistral (Mistral AI) - fast and efficient
- Gemma (Google) - lightweight
- Phi-3 (Microsoft) - small but capable
- And 100+ more models

## Implementation

### Files Modified

**1. src/data/embedders.py**

Added support for both providers via `llm_provider` parameter:

```python
from llama_index.llms.ollama import Ollama

class EmbeddingManager:
    def __init__(self,
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 llm_provider: Literal["gemini", "ollama"] = "gemini",
                 llm_model: str = None,
                 temperature: float = 0.1):

        # Set defaults based on provider
        if llm_model is None:
            llm_model = "models/gemini-2.5-flash" if llm_provider == "gemini" else "qwen2.5:7b"

        # Initialize embeddings (same for both)
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

        # Initialize LLM based on provider
        if llm_provider == "gemini":
            # API-based (requires API key)
            api_key = os.getenv("GOOGLE_API_KEY")
            self.llm = Gemini(
                model=llm_model,
                api_key=api_key,
                temperature=temperature
            )

        elif llm_provider == "ollama":
            # Local (no API key needed)
            self.llm = Ollama(
                model=llm_model,
                request_timeout=300.0,  # 5 minutes for local inference
                temperature=temperature
            )
```

**2. src/rag/pipeline.py**

Added `llm_provider` parameter to pipeline initialization:

```python
class RAGPipeline:
    def __init__(self,
                 use_reranking: bool = False,
                 use_hybrid: bool = False,
                 use_query_expansion: bool = False,
                 llm_provider: str = "gemini"):  # NEW PARAMETER

        self.llm_provider = llm_provider
        # ... rest of initialization

    def load_existing(self, persist_dir: str = "data/vector_stores/faiss"):
        # Pass provider to embedding manager
        self.embed_manager = EmbeddingManager(llm_provider=self.llm_provider)
        # ... rest of loading
```

**3. scripts/compare_gemini_vs_ollama.py** (NEW)

Comprehensive comparison test that measures:
- Response time
- Answer quality
- Cost implications
- Privacy considerations
- Rate limits

## Installation

### 1. Install Ollama Platform

```bash
# Ubuntu/Debian
sudo snap install ollama

# Or use the install script
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Download a Model

```bash
# Qwen 2.5 7B (recommended for technical content)
ollama pull qwen2.5:7b

# Or try other models
ollama pull llama3.1:8b     # Meta's Llama
ollama pull mistral:7b      # Mistral AI
ollama pull qwen2.5:1.5b    # Smaller, faster Qwen
```

### 3. Install LlamaIndex Integration

```bash
cd /path/to/powergrid-ai-tutor
source venv/bin/activate
pip install llama-index-llms-ollama
```

### 4. Verify Installation

```bash
# Check Ollama is running
ollama list

# Should show:
# NAME          ID              SIZE      MODIFIED
# qwen2.5:7b    845dbda0ea48    4.7 GB    X minutes ago
```

## Usage

### Basic Usage

```python
from src.rag.pipeline import RAGPipeline

# Use Gemini (API-based)
pipeline_gemini = RAGPipeline(llm_provider="gemini")
pipeline_gemini.load_existing()
answer = pipeline_gemini.query("What is solar energy?")

# Use Ollama (local)
pipeline_ollama = RAGPipeline(llm_provider="ollama")
pipeline_ollama.load_existing()
answer = pipeline_ollama.query("What is solar energy?")
```

### With Full Pipeline

```python
# Full pipeline with Ollama (no API costs!)
pipeline = RAGPipeline(
    use_query_expansion=True,
    use_hybrid=True,
    use_reranking=True,
    llm_provider="ollama"  # Use local model
)
pipeline.load_existing()
answer = pipeline.query("Explain MPPT algorithms")
```

### Custom Model

```python
from src.data.embedders import EmbeddingManager

# Use different Ollama model
embed_manager = EmbeddingManager(
    llm_provider="ollama",
    llm_model="llama3.1:8b"  # Use Llama instead of Qwen
)
```

## Performance Comparison

### Test Setup
- **Query:** "What are the main advantages of solar energy?"
- **Hardware:** Dell Latitude 7480 (typical laptop)
- **Models:**
  - Gemini: models/gemini-2.5-flash (API)
  - Ollama: qwen2.5:7b (local)

### Results

| Metric | Gemini (API) | Ollama (Local) | Winner |
|--------|--------------|----------------|--------|
| **Response Time** | 2.15s | 38.56s | Gemini (18x faster) |
| **Answer Quality** | Excellent | Very Good | Gemini (slightly better) |
| **Cost per 1000 queries** | $3.00 | $0.00 | Ollama (free) |
| **Rate Limits** | 15-20/min | Unlimited | Ollama |
| **Privacy** | Data sent to Google | 100% private | Ollama |
| **Setup Complexity** | Easy (API key) | Medium (install + download) | Gemini |
| **Hardware Requirements** | None | 8GB+ RAM recommended | Gemini |

### Actual Answers

**Gemini Response (2.15s):**
> "Solar energy is free, practically inexhaustible, and truly renewable. Its conversion into electricity involves no polluting residues or greenhouse gas emissions."

**Ollama Response (38.56s):**
> "The main advantages of solar energy include its renewability and abundance, as it is free and practically inexhaustible. Additionally, solar energy production does not involve polluting residues or greenhouse gas emissions, making it an environmentally friendly source of power."

**Analysis:**
- Both answers are accurate and well-sourced
- Ollama's answer is slightly more detailed
- Gemini is more concise
- Quality difference is minimal for this query

## When to Use Each Option

### Use Gemini (API) When:

✅ **Speed is critical**
- Need responses in 2-3 seconds
- Real-time user interactions
- Production applications with SLA requirements

✅ **Best quality needed**
- High-stakes applications
- Complex reasoning required
- Need most accurate answers

✅ **Limited local resources**
- Running on low-RAM machine
- Cloud deployment (no local GPU/CPU)
- Serverless architecture

✅ **Low query volume**
- <1000 queries/month (cheap)
- Infrequent usage
- Development/testing

### Use Ollama (Local) When:

✅ **Cost is a concern**
- High query volume (>10,000/month)
- Budget constraints
- Want to avoid ongoing API costs

✅ **Privacy is critical**
- Sensitive data (medical, financial, legal)
- Compliance requirements (GDPR, HIPAA)
- Corporate policy against cloud AI

✅ **No rate limits needed**
- Batch processing
- Load testing
- High-throughput applications
- Development with unlimited testing

✅ **Offline operation required**
- No internet connectivity
- Air-gapped environments
- Edge deployments

✅ **Have good hardware**
- 16GB+ RAM available
- Modern CPU (or GPU for speed)
- Can tolerate 30-60s response times

## Cost Analysis

### Gemini API Costs

**Pricing (Gemini 2.5 Flash):**
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens

**Per Query Cost (Average RAG query):**
- Retrieval context: ~1,500 tokens input
- Generated answer: ~500 tokens output
- **Total: ~$0.003 per query**

**Monthly Cost Estimates:**
| Queries/Month | Cost |
|---------------|------|
| 100 | $0.30 |
| 1,000 | $3.00 |
| 10,000 | $30.00 |
| 100,000 | $300.00 |

### Ollama Costs

**One-time:**
- Download model: Free (4.7GB bandwidth)
- Storage: Free (need 4.7GB disk space)

**Ongoing:**
- API calls: $0 (no API)
- Electricity: ~$0.001 per query (CPU usage)
- **Total: Effectively $0**

**Break-even Analysis:**
- Gemini cost: $3.00 per 1,000 queries
- Ollama setup time: ~30 minutes
- **Break-even: ~1,000 queries** (if your time is worth $100/hour)

## Speed Optimization for Ollama

### Use Smaller Models

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
    llm_model="qwen2.5:1.5b"  # Faster responses
)
```

**Speed Comparison:**
- qwen2.5:7b - 38s (best quality)
- qwen2.5:1.5b - ~15s (good quality)
- phi3:mini - ~8s (decent quality)

### GPU Acceleration

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

## Hybrid Approach: Best of Both Worlds

### Strategy 1: Development vs Production

```python
import os

# Use Ollama for development (unlimited testing)
# Use Gemini for production (best quality + speed)
llm_provider = "gemini" if os.getenv("ENVIRONMENT") == "production" else "ollama"

pipeline = RAGPipeline(
    use_hybrid=True,
    use_reranking=True,
    llm_provider=llm_provider
)
```

### Strategy 2: Tiered Service

```python
def get_pipeline(user_tier: str):
    if user_tier == "premium":
        # Premium users get fastest, best quality
        return RAGPipeline(llm_provider="gemini")
    else:
        # Free users get local (still good!)
        return RAGPipeline(llm_provider="ollama")
```

### Strategy 3: Fallback System

```python
def query_with_fallback(question: str):
    try:
        # Try Gemini first (fast + quality)
        pipeline = RAGPipeline(llm_provider="gemini")
        return pipeline.query(question)
    except RateLimitError:
        # Fall back to Ollama if rate limited
        print("Rate limited, using local model...")
        pipeline = RAGPipeline(llm_provider="ollama")
        return pipeline.query(question)
```

## Troubleshooting

### Ollama Server Not Running

**Error:** `Connection refused` or `Unable to connect to Ollama`

**Solution:**
```bash
# Start Ollama server
ollama serve

# Or run in background
ollama serve > /dev/null 2>&1 &
```

### Model Not Found

**Error:** `Model 'qwen2.5:7b' not found`

**Solution:**
```bash
# Pull the model
ollama pull qwen2.5:7b

# List available models
ollama list
```

### Timeout Errors

**Error:** `Request timeout after 120s`

**Solution:** Already handled in our code (300s timeout), but you can increase if needed:

```python
# In embedders.py, increase timeout
self.llm = Ollama(
    model=llm_model,
    request_timeout=600.0,  # 10 minutes
    temperature=temperature
)
```

### Out of Memory

**Error:** System freezes or OOM errors

**Solution:**
```bash
# Use smaller model
ollama pull qwen2.5:1.5b  # Only 1GB RAM needed

# Or use quantized version
ollama pull qwen2.5:7b-q4_0  # 4-bit quantized, 50% less RAM
```

## Future Improvements

### 1. Model Caching

Pre-load model to avoid first-query slowness:

```python
# Warm up Ollama on startup
if llm_provider == "ollama":
    print("Warming up local model...")
    Settings.llm.complete("Test")
    print("Model ready!")
```

### 2. Batch Processing

Process multiple queries efficiently with Ollama:

```python
# Process overnight with Ollama (no rate limits)
for question in large_question_list:
    answer = pipeline_ollama.query(question)
    save_to_database(question, answer)
```

### 3. Multi-Model Ensemble

Combine predictions from multiple models:

```python
# Get answers from both
answer_gemini = pipeline_gemini.query(question)
answer_ollama = pipeline_ollama.query(question)

# Use Gemini's answer, but log if they disagree significantly
if similarity(answer_gemini, answer_ollama) < 0.7:
    log_discrepancy(question, answer_gemini, answer_ollama)
```

### 4. Fine-Tuning Local Models

Fine-tune Ollama models on your specific domain:

```bash
# Create custom model fine-tuned on power systems
ollama create powergrid-expert -f Modelfile

# Use in pipeline
pipeline = RAGPipeline(
    llm_provider="ollama",
    llm_model="powergrid-expert"
)
```

## Conclusion

The Ollama integration provides a powerful **free, private, unlimited** alternative to API-based LLMs.

**Key Takeaways:**

1. **Gemini is faster (18x)** but costs money and has rate limits
2. **Ollama is free** after setup and has unlimited usage
3. **Both produce good quality answers** - Ollama is surprisingly close to Gemini
4. **Best approach:** Use Ollama for development, Gemini for production (or vice versa based on priorities)

**Recommended Default:**
- **Development:** Ollama (unlimited testing, no costs)
- **Production (low traffic):** Gemini (fast, reliable)
- **Production (high traffic):** Ollama with GPU (cost-effective)

The system now supports both, giving you flexibility to choose based on your specific needs!

## Related Documentation

- [Phase 5a: Query Expansion](phase5a-query-expansion.md)
- [Phase 4b: Hybrid Search](phase4b-hybrid-search.md)
- [FAQs: Index Rebuild and FAISS Limitations](faqs-index-rebuild-and-faiss-limitations.md)
