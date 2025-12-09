"""
Gradio UI for PowerGrid AI Tutor - HuggingFace Spaces version with API key input.
"""

import sys
import os
from pathlib import Path
import gradio as gr

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import time
from src.rag.pipeline import RAGPipeline

class PowerGridTutorUI:
 """
 Gradio interface for the PowerGrid AI Tutor with API key input.
 """
 
 logger = logging.getLogger(__name__)
 logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
 
 def __init__(self,
 use_reranking: bool = True,
 use_hybrid: bool = True,
 use_query_expansion: bool = True,
 llm_provider: str = "gemini"):
 """
 Initialize UI (pipeline created lazily after API key provided).
 """
 self.use_reranking = use_reranking
 self.use_hybrid = use_hybrid
 self.use_query_expansion = use_query_expansion
 self.llm_provider = llm_provider
 self.pipeline = None
 self.available_sources = []
 
 def initialize_pipeline(self, api_key: str, provider: str = "gemini", selected_features: list = None):
 """Initialize the RAG pipeline with the provided API key, provider, and selected features."""
 if not api_key or not api_key.strip():
 return f"ERROR: Please provide a valid {provider.upper()} API key"
 
 try:
 # Set environment variable based on provider
 if provider == "gemini":
 os.environ['GOOGLE_API_KEY'] = api_key.strip()
 elif provider == "openai":
 os.environ['OPENAI_API_KEY'] = api_key.strip()
 
 # Update the provider
 self.llm_provider = provider
 
 # Parse selected features
 if selected_features is None:
 selected_features = []
 use_query_expansion = "Query Expansion" in selected_features
 use_hybrid = "Hybrid Search" in selected_features
 use_reranking = "Reranking" in selected_features
 
 # Initialize pipeline
 self.pipeline = RAGPipeline(
 use_reranking=use_reranking,
 use_hybrid=use_hybrid,
 use_query_expansion=use_query_expansion,
 llm_provider=provider,
 api_key=api_key.strip()
 )
 
 # Load existing vector store
 self.pipeline.load_existing(persist_dir="data/vector_stores/faiss_full")
 
 # Get available sources
 self.available_sources = self._get_available_sources()
 
 features = []
 if use_query_expansion:
 features.append("Query Expansion")
 if use_hybrid:
 features.append("Hybrid Search")
 if use_reranking:
 features.append("Reranking")
 
 features_str = " + ".join(features) if features else "Basic RAG"
 return f"PowerGrid AI Tutor initialized successfully! ({self.llm_provider.upper()}: {features_str})"
 
 except Exception as e:
 return f"ERROR: Failed to initialize: {str(e)}"
 
 def _get_available_sources(self):
 """Get unique source files from the vector index."""
 if not self.pipeline:
 return []
 try:
 nodes = self.pipeline.index.docstore.docs.values()
 sources = set()
 for node in nodes:
 if hasattr(node, 'metadata') and 'source' in node.metadata:
 sources.add(node.metadata['source'])
 return sorted(list(sources))
 except Exception as e:
 print(f"Warning: Could not retrieve sources: {e}")
 return []
 
 def chat(self, message, history, topic_filter, source_type_filter):
 """Handle chat messages with optional metadata filtering."""
 if not self.pipeline:
 return history + [{"role": "assistant", "content": "Please initialize the system with your API key first."}], message, gr.update()
 
 if not message.strip():
 return history, message, gr.update()

 # Build filters dictionary
 filters = {}
 if topic_filter and topic_filter != "All Topics":
 filters['topic'] = topic_filter.lower()

 try:
 # Start timing
 import time
 start_time = time.time()
 
 # Get answer from RAG pipeline with filters
 result = self.pipeline.query(message, filters=filters if filters else None, return_sources=True)
 
 # Calculate processing time
 processing_time = time.time() - start_time
 
 # Extract answer and sources
 if isinstance(result, dict):
 answer = result.get('answer', str(result))
 sources = result.get('sources', [])
 
 # Format response with sources
 response_text = answer
 
 # Only show sources if answer is meaningful (not "does not contain" or "don't have information")
 if sources and "don't have information" not in answer.lower():
 response_text += "\n\n**Sources:**\n"
 for i, src in enumerate(sources[:3], 1):
 # Handle both dict and NodeWithScore objects
 if isinstance(src, dict):
 source_file = src.get('source', 'Unknown')
 score = src.get('score', 0)
 else:
 # NodeWithScore object
 source_file = src.metadata.get('source', 'Unknown') if hasattr(src, 'metadata') else 'Unknown'
 score = src.score if hasattr(src, 'score') else 0
 
 # Format relevance score as percentage
 relevance_pct = f"{score * 100:.1f}%" if score > 0 else "N/A"
 response_text += f"{i}. {source_file} (relevance: {relevance_pct})\n"
 
 # Add processing time
 response_text += f"\n*Processing time: {processing_time:.2f}s*"
 else:
 response_text = str(result)
 response_text += f"\n\n*Processing time: {processing_time:.2f}s*"
 
 # Append to history
 history.append({"role": "user", "content": message})
 history.append({"role": "assistant", "content": response_text})
 
 return history, "", gr.update(interactive=False)
 except Exception as e:
 error_msg = str(e)
 
 # Provide user-friendly error messages
 if "response.text" in error_msg or "safety_ratings" in error_msg:
 user_friendly_msg = "Unable to generate a response. This question may be outside my knowledge domain or blocked by safety filters. Please try asking about electrical engineering, renewable energy, or power systems."
 elif "does not contain" in error_msg.lower() or "no information" in error_msg.lower():
 user_friendly_msg = "ℹ I couldn't find relevant information in my knowledge base to answer this question. Please ask about topics related to electrical engineering, renewable energy, power systems, or smart grids."
 elif "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
 user_friendly_msg = "API rate limit reached. Please wait a moment and try again."
 elif "api key" in error_msg.lower() or "authentication" in error_msg.lower():
 user_friendly_msg = "ERROR: API key error. Please check your API key and try reinitializing."
 else:
 user_friendly_msg = f"ERROR: An error occurred: {error_msg}"
 
 history.append({"role": "user", "content": message})
 history.append({"role": "assistant", "content": user_friendly_msg})
 return history, "", gr.update(interactive=False)
 
 def create_interface(self):
 """Create and return the Gradio interface with API key input."""

 with gr.Blocks() as interface:
 
 # Header
 gr.HTML("""
 <div class="header">
 <h1> PowerGrid AI Tutor</h1>
 <p>Your Electrical Engineering & Renewable Energy Assistant</p>
 <p style="font-size: 14px;">LLM: GEMINI | Knowledge: 50 papers, 2166 chunks | Features: Query Expansion + Hybrid Search + Reranking</p>
 </div>
 """)
 
 # RAG Feature Selection
 with gr.Accordion("Advanced RAG Features", open=True):
 gr.Markdown("Enable advanced retrieval features (each adds API calls):")
 feature_checkboxes = gr.CheckboxGroup(
 choices=[
 "Query Expansion",
 "Hybrid Search", 
 "Reranking"
 ],
 value=["Query Expansion", "Hybrid Search", "Reranking"],
 label="Select Features"
 )
 
 # API Key Input Section
 with gr.Row():
 provider_choice = gr.Dropdown(
 choices=["gemini", "openai"],
 value="gemini",
 label="LLM Provider",
 scale=1
 )
 api_key_input = gr.Textbox(
 label="API Key",
 placeholder="Enter your API key here...",
 type="password",
 scale=3
 )
 init_button = gr.Button("Initialize", scale=1, variant="primary")
 
 gr.Markdown("""
 **Get API Keys:** 
 - Gemini: [Google AI Studio](https://aistudio.google.com/app/apikey) (Free tier: 15 RPM)
 - OpenAI: [OpenAI Platform](https://platform.openai.com/api-keys) (Paid, no region restrictions)
 """)
 
 status_message = gr.Textbox(label="Status", interactive=False)
 
 # Initialize button handler
 init_button.click(
 fn=self.initialize_pipeline,
 inputs=[api_key_input, provider_choice, feature_checkboxes],
 outputs=[status_message]
 )
 
 gr.Markdown("---")
 
 # Metadata filters
 with gr.Row():
 topic_filter = gr.Dropdown(
 choices=["All Topics", "Solar", "Wind", "Battery", "Grid", "General"],
 value="All Topics",
 label="Filter by Topic",
 scale=1
 )
 
 # Document type filter with checkboxes
 with gr.Accordion("Filter by Source Document Type", open=False):
 source_type_filter = gr.CheckboxGroup(
 choices=["Research Papers", "Standards", "Technical Reports", "Books"],
 value=["Research Papers", "Standards", "Technical Reports", "Books"],
 label="Select document types to search"
 )

 # Main chat interface
 chatbot = gr.Chatbot(
 height=500,
 label="Chat with PowerGrid AI Tutor",
 show_label=True
 )

 # Input area
 with gr.Row():
 msg = gr.Textbox(
 placeholder="Ask me about solar energy, wind power, batteries, smart grids, or power systems...",
 show_label=False,
 scale=4
 )
 submit = gr.Button("Send", scale=1, variant="primary", interactive=False)
 
 # Example questions
 gr.Examples(
 examples=[
 "What are the main challenges in integrating solar power into the electrical grid?",
 "How does wind energy affect power grid stability?",
 "What are the latest advances in battery energy storage systems?",
 "Explain smart grid technology and its benefits",
 "What is the role of inverters in solar photovoltaic systems?"
 ],
 inputs=msg,
 label="Example Questions"
 )
 
 # Clear button
 clear = gr.Button("Clear Chat")
 
 # Event handlers
 def update_button_state(text):
 has_text = bool(text and text.strip())
 return gr.update(interactive=has_text)

 msg.input(fn=update_button_state, inputs=[msg], outputs=[submit])
 msg.change(fn=update_button_state, inputs=[msg], outputs=[submit])

 submit.click(
 fn=self.chat,
 inputs=[msg, chatbot, topic_filter, source_type_filter],
 outputs=[chatbot, msg, submit]
 )

 msg.submit(
 fn=self.chat,
 inputs=[msg, chatbot, topic_filter, source_type_filter],
 outputs=[chatbot, msg, submit]
 )

 clear.click(fn=lambda: [], outputs=[chatbot])
 
 # Footer
 gr.Markdown("""
 ---
 **About:** This AI tutor uses state-of-the-art Retrieval-Augmented Generation (RAG) to answer questions
 about electrical engineering and renewable energy based on 50 research papers.

 **Technology:** FAISS vector store | HuggingFace BAAI/bge-small-en-v1.5 embeddings | LlamaIndex | Metadata Filtering

 **Features Active:**
 - **Query Expansion:** LLM generates technical terms for better keyword matching (+10-20% accuracy)
 - **Hybrid Search:** BM25 (keywords) + Semantic (meaning) with RRF fusion (+5-15% accuracy)
 - **LLM Reranking:** LLM scores retrieved chunks for relevance (+15-25% accuracy)
 
 **Estimated Total Accuracy Gain:** +30-50% over baseline
 
 **Note:** You need a Google Gemini API key to use this app. Get one for free at [Google AI Studio](https://makersuite.google.com/app/apikey).
 """)
 
 return interface
