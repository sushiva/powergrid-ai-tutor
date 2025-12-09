---
title: PowerGrid AI Tutor
emoji: ⚡
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.0.2
app_file: app.py
pinned: false
license: mit
short_description: AI tutor for electrical engineering & renewables
---

# PowerGrid AI Tutor

An advanced Retrieval-Augmented Generation (RAG) system for electrical engineering and renewable energy education.

## Features

- 🔋 50 research papers on solar, wind, battery, and grid technologies
- 🔍 Advanced RAG: query expansion, hybrid search, LLM reranking
- 📊 Evaluated system: 70% hit rate, 55% MRR
- 🎯 Domain-specific for electrical engineering
- ⚙️ Configurable features via UI

## How to Use

1. Choose your LLM provider (Gemini or OpenAI)
2. Enter your API key:
   - **Gemini**: Get free key at [Google AI Studio](https://aistudio.google.com/app/apikey)
   - **OpenAI**: Get key at [OpenAI Platform](https://platform.openai.com/api-keys) (paid, no regional restrictions)
3. Select RAG features (query expansion, hybrid search, reranking)
4. Ask questions about renewable energy, power systems, or smart grids!

## Example Questions

- "How do wind turbines generate electricity?"
- "What are the latest advances in battery storage?"
- "Explain the challenges of integrating solar power into the grid"

## System Details

- **Knowledge Base**: 50 ArXiv papers (852 pages, 2,166 chunks)
- **Embedding Model**: BAAI/bge-small-en-v1.5 (local, no API cost)
- **LLM**: Google Gemini 2.5 Flash (~$0.003/query)
- **Vector Store**: FAISS with semantic + keyword search

## Cost

- Average cost per query: $0.003
- Free tier: 15 requests/minute
- Demo cost: < $0.10 (well under $0.50 requirement)

Built for LLM Developer Certification - Advanced RAG Project
# Trigger rebuild
