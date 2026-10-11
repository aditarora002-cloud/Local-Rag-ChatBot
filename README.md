Just shipped my first proper end-to-end AI project — a fully local, offline RAG chatbot that chats with your own PDFs, Word docs (.docx) & PowerPoint slides (.pptx) 🔥

No APIs, no cloud costs, everything runs on my laptop using open-source tools.

What it does:
• Loads & chunks documents intelligently
• Stores embeddings locally with ChromaDB
• Retrieves relevant parts + generates grounded answers with Llama 3.2 (via Ollama)
• Remembers conversation + shows exact sources (page/file name)
• Simple Streamlit UI for chatting

Biggest win? When I fed it my college notes + a random assignment .docx and asked complex questions — it quoted the exact lines correctly. That “it actually works” moment was pure joy 😭

Struggles → victories:
- Fixing Ollama port bind errors
- Tuning chunk size/overlap for better retrieval
- Adding support for .docx & .pptx (Docx2txt + Unstructured)
- Adding memory so it doesn't forget context

Learned: RAG is not just hype — it's the practical way to make LLMs useful without sending your data to the cloud.
