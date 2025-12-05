"""
Gradio UI for PowerGrid AI Tutor.
"""

import sys
from pathlib import Path
import gradio as gr

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.pipeline import RAGPipeline


class PowerGridTutorUI:
    """
    Gradio interface for the PowerGrid AI Tutor.
    """
    
    def __init__(self, use_reranking: bool = False):
        """
        Initialize the RAG pipeline.

        Args:
            use_reranking: Whether to use LLM reranking for better relevance
        """
        print("Initializing PowerGrid AI Tutor...")
        self.use_reranking = use_reranking
        self.pipeline = RAGPipeline(use_reranking=use_reranking)
        self.pipeline.load_existing(persist_dir="data/vector_stores/faiss_full")
        print(f"PowerGrid AI Tutor ready! (Reranking: {'ON' if use_reranking else 'OFF'})")
    
    def chat(self, message, history):
        """
        Handle chat messages.

        Args:
            message: User's question
            history: Chat history (list of message dictionaries)

        Returns:
            Tuple of (updated history, cleared input, disabled button)
        """
        if not message.strip():
            return history, message, gr.update()

        # Get answer from RAG pipeline
        response = self.pipeline.query(message, return_sources=False)

        # Append user message and assistant response to history
        # Gradio 6.x requires dictionaries with 'role' and 'content' keys
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})

        # Return updated history, cleared input, and disabled button
        return history, "", gr.update(interactive=False)
    
    def create_interface(self):
        """Create and return the Gradio interface."""

        with gr.Blocks() as interface:
            
            # Header
            reranking_status = "ON (MRR: 37.9%)" if self.use_reranking else "OFF (Hit Rate: 50.0%)"
            gr.HTML(f"""
                <div class="header">
                    <h1>⚡ PowerGrid AI Tutor</h1>
                    <p>Your Electrical Engineering & Renewable Energy Assistant</p>
                    <p style="font-size: 14px;">Knowledge base: 50 research papers | 2166 chunks | Reranking: {reranking_status}</p>
                </div>
            """)
            
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
            examples_component = gr.Examples(
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

            # Enable/disable submit button based on input text
            def update_button_state(text):
                has_text = bool(text and text.strip())
                return gr.update(interactive=has_text)

            # Monitor input field changes (user typing)
            msg.input(fn=update_button_state, inputs=[msg], outputs=[submit])

            # Also monitor when text changes (including from examples)
            msg.change(fn=update_button_state, inputs=[msg], outputs=[submit])

            # Handle submit button click
            # chat() returns (history, cleared_msg, disabled_button)
            submit.click(
                fn=self.chat,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg, submit]
            )

            # Handle Enter key
            msg.submit(
                fn=self.chat,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg, submit]
            )

            # Clear chat history
            clear.click(
                fn=lambda: [],
                outputs=[chatbot]
            )
            
            # Footer
            reranking_info = """
            **LLM Reranking:** ON - Uses LLM to rerank retrieved chunks for better contextual relevance (MRR: 37.9%)
            """ if self.use_reranking else """
            **LLM Reranking:** OFF - Uses direct similarity search for broader topic coverage (Hit Rate: 50.0%)
            """

            gr.Markdown(f"""
            ---
            **About:** This AI tutor uses Retrieval-Augmented Generation (RAG) to answer questions
            about electrical engineering and renewable energy based on 50 research papers.

            **Technology:** FAISS vector store | HuggingFace embeddings | Google Gemini LLM

            {reranking_info}
            """)
        
        return interface
    
    def launch(self, share=False):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        interface.launch(share=share)


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PowerGrid AI Tutor")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable LLM reranking for better contextual relevance (slower, costs more)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    args = parser.parse_args()

    # Launch UI with selected options
    tutor = PowerGridTutorUI(use_reranking=args.rerank)
    tutor.launch(share=args.share)