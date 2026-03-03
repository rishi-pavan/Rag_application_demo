RAG Application Demo 🚀

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload a PDF and ask questions — powered by embeddings and a large language model.

This demo showcases how to combine vector search (FAISS) with contextual generation (Gemini / other LLMs) in a simple Streamlit UI.

🧠 Features

📄 PDF Upload & Text Extraction – Load any PDF and extract its contents.

🔍 Semantic Search – Chunk the extracted text and index it with FAISS using embeddings.

🧠 Context-Aware Responses – Combine semantic context with generative output from an LLM.

💬 Interactive UI – Streamlit-based interface for real-time Q&A.

🚀 Quick Demo (How It Works)

Upload a PDF file in the sidebar.

The app extracts text and splits it into chunks.

Each chunk is embedded and stored in a FAISS vector index.

When you ask a query, the most relevant chunks are retrieved.

The model (Gemini or configured LLM) generates a response based on retrieved context.