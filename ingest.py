from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# === CONFIG ===
DATA_FOLDER = "data"                  # your folder with PDFs, DOCX, PPTX
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

print("Starting ingestion...")

# 1. Load documents from multiple formats
documents = []

data_path = Path(DATA_FOLDER)

# PDFs
pdf_loader = PyPDFDirectoryLoader(str(data_path))
documents.extend(pdf_loader.load())

# DOCX
for docx_file in data_path.glob("**/*.docx"):
    loader = Docx2txtLoader(str(docx_file))
    documents.extend(loader.load())

# PPTX
for pptx_file in data_path.glob("**/*.pptx"):
    loader = UnstructuredPowerPointLoader(
        str(pptx_file),
        mode="elements",          # "elements" = better for slides/tables/notes
        strategy="fast"           # or "hi_res" if you install more deps
    )
    documents.extend(loader.load())

print(f"Loaded {len(documents)} documents (PDFs + DOCX + PPTX)")

if not documents:
    print("No documents found in data folder!")
    exit()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# 3. Embeddings
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# 4. Save to Chroma (overwrites old db)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)

print(f"Vector store updated and saved to: {CHROMA_PATH}")
print("Ingestion complete!")