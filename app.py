import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# === CONFIG ===
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
TOP_K = 5

@st.cache_resource
def load_components():
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL)
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    
    llm = ChatOllama(model=LLM_MODEL, temperature=0.6)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful, accurate assistant. 
Answer ONLY using the provided document context.
If the information isn't there, say clearly: "I don't have information about that in the documents."
Be concise unless asked for more detail."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        RunnableParallel(
            {
                "context": (lambda x: x["question"]) | retriever | format_docs,
                "question": lambda x: x["question"],
                "history": lambda x: x["history"]
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever

st.title("📄 Local RAG Chatbot (with memory & sources)")
st.caption("Ask questions about your PDFs – everything runs locally")

chain, retriever = load_components()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get answer
            response = chain.invoke({
                "question": prompt,
                "history": [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]  # exclude current prompt
                ]
            })

            st.markdown(response)

            # Show sources in expander
            docs = retriever.invoke(prompt)
            if docs:
                with st.expander("📄 Sources used"):
                    for i, doc in enumerate(docs, 1):
                        filename = doc.metadata.get("source", "unknown").split("\\")[-1].split("/")[-1]
                        page = doc.metadata.get("page", "?")
                        preview = doc.page_content[:350].replace("\n", " ") + "..." if len(doc.page_content) > 350 else doc.page_content
                        st.markdown(f"**Source {i}**  –  page {page}  –  {filename}")
                        st.text(preview)
            else:
                st.caption("No relevant sources retrieved.")

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Small footer
st.markdown("---")
st.caption("Local RAG • Ollama + Chroma • Built by Adit")