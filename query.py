from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage

# === CONFIG ===
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"          # change to your preferred model if needed
TOP_K = 5                          # feel free to adjust (4–8 usually good)

print("Loading vector store...")
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL)
)

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

llm = ChatOllama(model=LLM_MODEL, temperature=0.6)

# Better system prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful, accurate assistant. 
Answer ONLY using the provided context from the documents.
If the information isn't in the context, clearly say: "I don't have information about that in the documents."
Be concise unless more detail is requested.
Use bullet points or numbered lists when it improves clarity."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain with parallel retrieval for context + sources
rag_chain = (
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

print("\nImproved RAG chatbot ready! (with memory + sources)")
print("Type your questions below (type 'exit', 'quit' or 'q' to stop)\n")

history = []  # will store conversation messages

while True:
    question = input("You: ").strip()
    
    if question.lower() in ['exit', 'quit', 'q', '']:
        print("Goodbye!")
        break
    
    if not question:
        continue

    print("Thinking...")

    # Run the chain
    result = rag_chain.invoke({
        "question": question,
        "history": history
    })

    print(f"\nBot: {result}\n")

    # Show sources
    docs = retriever.invoke(question)
    if docs:
        print("Used sources:")
        for i, doc in enumerate(docs, 1):
            content_preview = doc.page_content.replace("\n", " ")[:180].strip() + "..."
            page = doc.metadata.get("page", "?")
            filename = doc.metadata.get("source", "unknown").split("\\")[-1].split("/")[-1]
            print(f"  [{i}]  page {page}  •  {filename}")
            print(f"      {content_preview}\n")
    else:
        print("No relevant sources found.\n")

    print("-" * 70)

    # Update conversation history
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=result))

    # Optional: limit history length to avoid context overflow
    if len(history) > 12:
        history = history[-12:]