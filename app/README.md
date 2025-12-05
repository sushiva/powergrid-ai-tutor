# PowerGrid AI Tutor - Gradio UI

Interactive chat interface for the PowerGrid AI Tutor RAG system.

## Running the UI

### Basic Mode (No Reranking - Faster)
```bash
python app/main.py
```

**Performance:**
- Hit Rate: 50.0%
- MRR: 33.9%
- Faster responses
- Better topic coverage

### Enhanced Mode (With LLM Reranking)
```bash
python app/main.py --rerank
```

**Performance:**
- Hit Rate: 45.0%
- MRR: 37.9%
- Slower responses (~2-3 seconds extra)
- Better contextual relevance
- Higher API costs

### Public Sharing
```bash
python app/main.py --share
```

Creates a public Gradio link to share with others.

## Features

- **Chat Interface**: Ask questions about electrical engineering and renewable energy
- **Example Questions**: Quick-start prompts for common topics
- **Smart Input**: Submit button enables only when text is entered
- **Clear Chat**: Reset conversation history
- **Configurable Reranking**: Choose between speed vs relevance

## Technology Stack

- **UI Framework**: Gradio 6.x
- **Vector Store**: FAISS
- **Embeddings**: HuggingFace (BAAI/bge-small-en-v1.5)
- **LLM**: Google Gemini 2.5 Flash
- **Knowledge Base**: 50 research papers (2,166 chunks)

## Reranking Comparison

| Mode | Hit Rate | MRR | Speed | Use Case |
|------|----------|-----|-------|----------|
| **No Reranking** | 50.0% | 33.9% | Fast | Broad topic queries |
| **With Reranking** | 45.0% | 37.9% | Slower | Precise, context-specific queries |

**Recommendation**: Use reranking for detailed technical questions, skip it for quick topic exploration.
