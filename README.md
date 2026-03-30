# 🤗 RAG Pipeline with HuggingFace + Multi-Document Support

A complete **Retrieval-Augmented Generation (RAG)** pipeline that supports multiple document types (PDF, Excel, Word) using HuggingFace embeddings and OpenRouter LLM API.

## 📋 Overview

**RAG Flow:**
```
Upload Docs → Split → Embed (HF) → FAISS Store → Retrieve → Augment → Generate (LLM)
```

This project enables you to:
- 📄 Upload and process **PDF**, **Excel (.xlsx)**, and **Word (.docx)** files
- 🔢 Generate embeddings using free HuggingFace models
- 🔍 Store and retrieve document chunks using FAISS vector store
- 🤖 Generate answers using LLMs via OpenRouter API
- 💬 Interactive Q&A against your document knowledge base

---

## ✨ Features

| Feature | Details |
|---------|---------|
| **Multi-Format Support** | PDF, Excel, Word documents |
| **Free Embeddings** | HuggingFace `BAAI/bge-small-en-v1.5` (no API key needed) |
| **Vector Store** | FAISS for fast similarity search |
| **LLM Integration** | Llama 3.3 70B via OpenRouter API |
| **Local Processing** | Embeddings run locally (CPU/GPU) |
| **Interactive Mode** | Real-time Q&A interface |
| **Source Tracking** | Know which document/page each answer comes from |

---

## 🔧 Prerequisites

### System Requirements
- Python 3.8+
- At least 4GB RAM (8GB recommended for embeddings)
- Internet connection (for LLM API)

### API Keys Required
1. **OpenRouter API Key** - Get free credits at https://openrouter.ai/keys
   - Used for LLM inference (Llama 3.3 70B)
   - Free tier available with limited requests

---

## 📦 Installation & Setup

### Step 1: Create Virtual Environment

```bash
# Navigate to project directory
cd "d:\LWP\GenAI projects for LWP\RAG Agents"

# Create virtual environment
python -m venv venvt

# Activate virtual environment
# On Windows:
venvt\Scripts\activate
# On macOS/Linux:
source venvt/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install all required libraries
pip install -q \
    langchain \
    langchain-community \
    langchain-huggingface \
    langchain-openai \
    faiss-cpu \
    sentence-transformers \
    huggingface-hub \
    pypdf \
    openpyxl \
    python-docx \
    docx2txt \
    python-dotenv

# Or use the upgrade command (recommended for latest versions)
pip install --upgrade \
    langchain==0.3.25 \
    langchain-core==0.3.58 \
    langchain-community==0.3.24 \
    langchain-openai \
    langchain-huggingface==0.1.2 \
    langchain-text-splitters==0.3.8 \
    faiss-cpu \
    sentence-transformers \
    pypdf \
    openpyxl \
    python-docx \
    docx2txt
```

**What each library does:**
- `langchain` - Framework for building RAG applications
- `langchain-community` - Additional integrations and tools
- `langchain-huggingface` - HuggingFace embeddings support
- `langchain-openai` - OpenRouter API integration
- `faiss-cpu` - Vector similarity search library
- `sentence-transformers` - Pre-trained embedding models
- `huggingface-hub` - Download models from HuggingFace
- `pypdf` - PDF document loading
- `openpyxl` - Excel file support
- `python-docx` - Word document support
- `docx2txt` - Extract text from Word files

### Step 3: Configure API Keys

Create a `.env` file in the project directory:

```bash
# .env file
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
```

---

## 📂 Project Structure

```
RAG Agents/
├── README.md                                    # This file
├── rag_agent_knowledge_sourrce.py               # Main Python script
├── RAG_for_Agent_Knowledge_Sourrce (4).ipynb    # Jupyter notebook version
├── venvt/                                        # Virtual environment
├── .env                                         # API keys (add this file)
└── [Your Documents]/
    ├── document.pdf
    ├── data.xlsx
    └── report.docx
```

---

## 🚀 Step-by-Step Execution Guide

### **Step 0: Set API Key**

```python
import os
os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-your-key-here"
print("✅ OpenRouter API key set!")
```

**Purpose:** Authenticates requests to OpenRouter API for LLM inference.

**How it works:**
- Loads your OpenRouter API key into environment
- Used later when initializing the ChatOpenAI client
- Get free key from: https://openrouter.ai/keys

---

### **Step 1a: Import Libraries**

```python
import os
import tempfile
import pandas as pd

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document
from langchain_core.documents import Document

# Embeddings and LLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI

# Vector Store
from langchain_community.vectorstores import FAISS

# Prompts and Runnables
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Document Loaders
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

print("✅ Imports done!")
```

**What each import does:**
- `RecursiveCharacterTextSplitter` - Chunking text while preserving context
- `Document` - Represents a text chunk with metadata
- `HuggingFaceEmbeddings` - Free embedding model
- `ChatOpenAI` - Interface to OpenRouter API
- `FAISS` - Vector storage and search
- `PromptTemplate` - Structure for prompts to LLM
- `Runnable*` - Chain components together
- `Docx2txtLoader`, `PyPDFLoader` - Load different document formats

---

### **Step 1b: Document Ingestion & Loading**

#### Configure File Paths

```python
FILE_PATHS = [
    "/path/to/your/document.pdf",
    # "/path/to/your/data.xlsx",
    # "/path/to/your/report.docx",
]
```

**Instructions:**
1. Place your documents in the project directory or any accessible folder
2. Update `FILE_PATHS` with actual file paths
3. Supports relative and absolute paths
4. Uncomment/add lines for files you want to include

#### Load Documents Function

```python
def load_documents(file_paths: list) -> list:
    """Load PDF, Excel (.xlsx), and Word (.docx) files into LangChain Documents."""
    all_docs = []
    
    for path in file_paths:
        ext = os.path.splitext(path)[-1].lower()
        print(f"📄 Loading: {path}  [{ext}]")
        
        # Load PDF files
        if ext == ".pdf":
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"   ✅ Loaded {len(docs)} pages from PDF")
        
        # Load Excel files
        elif ext in [".xlsx", ".xls"]:
            # Read all sheets
            xls = pd.read_excel(path, sheet_name=None)
            for sheet_name, df in xls.items():
                text = f"Sheet: {sheet_name}\n" + df.to_string(index=False)
                doc = Document(
                    page_content=text,
                    metadata={"source": path, "sheet": sheet_name}
                )
                all_docs.append(doc)
            print(f"   ✅ Loaded {len(xls)} sheet(s) from Excel")
        
        # Load Word documents
        elif ext == ".docx":
            loader = Docx2txtLoader(path)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"   ✅ Loaded {len(docs)} section(s) from Word")
        
        # Skip unsupported formats
        else:
            print(f"   ⚠️  Unsupported format: {ext} — skipping")
    
    print(f"\n📦 Total documents loaded: {len(all_docs)}")
    return all_docs

# Execute document loading
raw_docs = load_documents(FILE_PATHS)

# Preview first document
if raw_docs:
    print("\n--- Preview of first document ---")
    print(raw_docs[0].page_content[:500])
    print("Metadata:", raw_docs[0].metadata)
else:
    print("⚠️  No files loaded. Add file paths to FILE_PATHS above and re-run.")
```

**What this does:**
- Iterates through each file in `FILE_PATHS`
- **PDF**: Uses `PyPDFLoader` to extract text page-by-page
- **Excel**: Reads all sheets using pandas, converts to Document objects
- **Word**: Uses `Docx2txtLoader` to extract paragraphs
- Returns list of `Document` objects with content and metadata
- Each document remembers its source (filename, page, sheet)

**Output:** List of documents ready for chunking

---

### **Step 1c: Text Splitting / Chunking**

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Each chunk is ~1000 characters
    chunk_overlap=200       # Chunks overlap by 200 chars (preserve context)
)

chunks = splitter.split_documents(raw_docs)

print(f"✅ Total chunks created: {len(chunks)}")

# Preview a chunk
if chunks:
    print("\n--- Sample Chunk ---")
    print(chunks[0].page_content)
    print("Metadata:", chunks[0].metadata)
```

**Why chunking?**
- Documents are too large for embeddings (LLMs have token limits)
- Smaller chunks = more specific retrieval results
- Overlap preserves context at chunk boundaries

**Parameters explained:**
- `chunk_size=1000`: Size of each chunk in characters
  - Smaller (500) = more chunks, more fine-grained retrieval
  - Larger (2000) = fewer chunks, broader context
- `chunk_overlap=200`: Overlap between consecutive chunks
  - Prevents losing information at boundaries
  - Higher overlap = more redundancy but better continuity

**Output:** Smaller text chunks with preserved metadata

---

### **Step 1d: Embedding Generation**

```python
# Use a free embedding model
embedding_model_name = "BAAI/bge-small-en-v1.5"
# Alternative models:
# "sentence-transformers/all-MiniLM-L6-v2" (very lightweight)
# "sentence-transformers/all-mpnet-base-v2" (higher quality, slower)

print("⏳ Loading embedding model (downloads once)...")

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "cpu"},     # Use "cuda" for GPU
    encode_kwargs={"normalize_embeddings": True}
)

print("✅ Embedding model loaded!")
```

**What this does:**
- Downloads pre-trained embedding model from HuggingFace
- `BAAI/bge-small-en-v1.5`: 110M parameters, good speed/quality balance
- Runs locally on your CPU/GPU (no API call for embeddings)
- `normalize_embeddings=True`: Optimizes for similarity search

**Why embeddings?**
- Convert text to vectors (numerical representation)
- Similar documents have similar vectors
- Enables fast similarity search in FAISS

---

### **Step 1e: Vector Store Creation (FAISS)**

```python
print("⏳ Building FAISS vector store...")
vector_store = FAISS.from_documents(chunks, embeddings)
print(f"✅ Vector store built! Total vectors: {vector_store.index.ntotal}")

# Optional: Save vector store to disk for reuse
# vector_store.save_local("faiss_index")
# Later, reload with:
# vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

print("Vector store ready. Uncomment above lines to persist to disk.")
```

**What FAISS does:**
- **F**acebook **A**rtificial **I**ntelligence **S**imilarity **S**earch
- Fast vector similarity search library
- `from_documents()`: Embeds all chunks and stores in index
- `vector_store.index.ntotal`: Total number of vectors stored

**Persistence (optional):**
- Large vector stores can be saved to disk
- Reloading is faster than re-embedding everything
- Useful for frequently-used document sets

---

### **Step 2: Create Retriever**

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}   # Retrieve top-4 most relevant chunks
)

print("✅ Retriever ready!")

# Test the retriever
test_query = "What is this document about?"
test_results = retriever.invoke(test_query)

print(f"Query: '{test_query}'")
print(f"Retrieved {len(test_results)} chunks:\n")
for i, doc in enumerate(test_results):
    print(f"[{i+1}] Source: {doc.metadata.get('source', 'N/A')} | Page: {doc.metadata.get('page', 'N/A')}")
    print(doc.page_content[:300])
    print("---")
```

**What this does:**
- Wraps FAISS in a retriever interface
- `search_type="similarity"`: Find most similar vectors
- `k=4`: Return top 4 most relevant chunks
- Test query shows how retrieval works

**Tuning `k` parameter:**
- `k=2`: Fast, concise (use for short questions)
- `k=4`: Balanced (default, recommended)
- `k=8`: Comprehensive, more context (use for complex questions)

**Output:** Relevant document chunks ranked by similarity

---

### **Step 3: Initialize LLM (OpenRouter)**

```python
from langchain_openai import ChatOpenAI

os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-your-key-here"

llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",  # Model ID on OpenRouter
    temperature=0.2,        # Lower = more deterministic (0-1)
    max_tokens=512,         # Max output length
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

print("✅ LLM ready!")

# Quick test
response = llm.invoke("Say hello in one sentence.")
print(response.content)
```

**Parameters explained:**
- `model`: LLM to use on OpenRouter
  - `meta-llama/llama-3.3-70b-instruct` (recommended, high quality)
  - `gpt-3.5-turbo`: Faster, cheaper alternative
  - Other available models: Check OpenRouter website
- `temperature=0.2`: Creativity level
  - 0.0 = Deterministic (good for FAQs, factual Q&A)
  - 0.5 = Balanced
  - 1.0 = Creative (good for brainstorming)
- `max_tokens=512`: Maximum response length
  - Smaller = faster, cheaper
  - Larger = longer answers

**Available models on OpenRouter:**
```python
# High quality (slower, more expensive)
model="meta-llama/llama-3.3-70b-instruct"

# Balanced (faster, cheaper)
model="gpt-3.5-turbo"

# Lightweight (fastest, cheapest)
model="mistralai/mistral-7b-instruct-v0.1"
```

---

### **Step 4: Create Prompt Template**

```python
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the following context:
    {context}

    Question: {question}
    """
)

print("✅ Prompt template ready!")
```

**What this does:**
- Defines the structure for LLM prompts
- `{context}`: Placeholder for retrieved document chunks
- `{question}`: Placeholder for user question
- Template ensures consistent prompt formatting

**Advanced prompt example:**
```python
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. Answer the question using ONLY the context provided below.
    If the answer is not in the context, say "I don't have enough information in the provided documents."
    Be concise and clear.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
)
```

---

### **Step 5a: Format Retrieved Documents**

```python
def format_docs(retrieved_docs):
    """Combine retrieved chunks into a single context string with source info."""
    parts = []
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "")
        sheet  = doc.metadata.get("sheet", "")
        
        # Build label with document details
        label = f"[Doc {i+1} | {os.path.basename(source)}"
        if page: 
            label += f" | Page {page}"
        if sheet: 
            label += f" | Sheet: {sheet}"
        label += "]"
        
        parts.append(f"{label}\n{doc.page_content}")
    
    return "\n\n".join(parts)

# Example output:
# [Doc 1 | document.pdf | Page 1]
# AI is the simulation of human intelligence...
# 
# [Doc 2 | data.xlsx | Sheet: Summary]
# Model Performance Metrics: Accuracy 95%...
```

**What this does:**
- Combines multiple chunks into a single context string
- Adds source labels (filename, page, sheet)
- Helps LLM know where information comes from
- Makes it easier to cite sources in answers

---

### **Step 5b: Build RAG Chain**

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# Parallel processing: retrieve docs AND pass question
parallel_chain = RunnableParallel({
    "context":  retriever | RunnableLambda(format_docs),  # Get context
    "question": RunnablePassthrough()                       # Keep question
})

# Full RAG chain
main_chain = parallel_chain | prompt | llm | parser

print("✅ RAG chain ready!")
```

**Chain flow diagram:**
```
User Question
    ↓
RunnableParallel (2 paths):
    ├→ retriever → format_docs → "context"
    └→ RunnablePassthrough() → "question"
    ↓
Merge into: {"context": "...text...", "question": "...text..."}
    ↓
prompt (template fills in context and question)
    ↓
llm (LLM generates response)
    ↓
parser (convert to plain text)
    ↓
Answer
```

**Component roles:**
- `retriever`: Finds relevant document chunks
- `format_docs`: Formats chunks with source info
- `prompt`: Templates the context + question
- `llm`: Generates answer
- `parser`: Converts response to text

---

### **Step 6: Interactive Q&A Loop**

```python
print("🤖 Interactive RAG Q&A")
print("Type your question below. Type 'exit' to stop.\n")

while True:
    user_input = input("You: ").strip()
    
    # Exit conditions
    if user_input.lower() in ["exit", "quit", "q"]:
        print("👋 Exiting. Goodbye!")
        break
    
    # Skip empty inputs
    if not user_input:
        continue
    
    print("⏳ Thinking...")
    
    # Invoke RAG chain
    response = main_chain.invoke(user_input)
    
    print(f"\n🤖 Answer: {response}\n")
    print("-" * 60)
```

**Flow for each question:**
1. User types question
2. Retriever searches vector store for relevant chunks
3. Top-4 chunks are formatted with source info
4. Question + context fed to LLM via template
5. LLM generates answer using context
6. Answer printed with source citation

**Example interaction:**
```
🤖 Interactive RAG Q&A
Type your question below. Type 'exit' to stop.

You: What is overfitting?
⏳ Thinking...

🤖 Answer: Based on the documents, overfitting occurs when a machine learning model learns the training data too well, including its noise and peculiarities, leading to poor performance on new data.
```

---

## 🔄 Complete End-to-End Code

Here's the full script in one block:

```python
# ============================================
# RAG PIPELINE - COMPLETE SETUP
# ============================================

import os
import tempfile
import pandas as pd

# ─────────────────────────────────────────────
# STEP 0: API KEY SETUP
# ─────────────────────────────────────────────
os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-your-key-here"
print("✅ OpenRouter API key set!")

# ─────────────────────────────────────────────
# STEP 1: IMPORTS
# ─────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_openai import ChatOpenAI

print("✅ Imports done!")

# ─────────────────────────────────────────────
# STEP 2: CONFIGURE FILES
# ─────────────────────────────────────────────
FILE_PATHS = [
    "path/to/your/document.pdf",
    # "path/to/your/data.xlsx",
    # "path/to/your/report.docx",
]

# ─────────────────────────────────────────────
# STEP 3: LOAD DOCUMENTS
# ─────────────────────────────────────────────
def load_documents(file_paths: list) -> list:
    """Load PDF, Excel, and Word files."""
    all_docs = []
    
    for path in file_paths:
        ext = os.path.splitext(path)[-1].lower()
        print(f"📄 Loading: {path}  [{ext}]")
        
        if ext == ".pdf":
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"   ✅ Loaded {len(docs)} pages")
        
        elif ext in [".xlsx", ".xls"]:
            xls = pd.read_excel(path, sheet_name=None)
            for sheet_name, df in xls.items():
                text = f"Sheet: {sheet_name}\n" + df.to_string(index=False)
                doc = Document(
                    page_content=text,
                    metadata={"source": path, "sheet": sheet_name}
                )
                all_docs.append(doc)
            print(f"   ✅ Loaded {len(xls)} sheets")
        
        elif ext == ".docx":
            loader = Docx2txtLoader(path)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"   ✅ Loaded {len(docs)} sections")
        
        else:
            print(f"   ⚠️  Unsupported: {ext}")
    
    print(f"\n📦 Total documents: {len(all_docs)}")
    return all_docs

raw_docs = load_documents(FILE_PATHS)

# ─────────────────────────────────────────────
# STEP 4: SPLIT TEXT INTO CHUNKS
# ─────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(raw_docs)
print(f"✅ Created {len(chunks)} chunks")

# ─────────────────────────────────────────────
# STEP 5: CREATE EMBEDDINGS & VECTOR STORE
# ─────────────────────────────────────────────
embedding_model_name = "BAAI/bge-small-en-v1.5"

print("⏳ Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("✅ Embedding model loaded!")

print("⏳ Building FAISS vector store...")
vector_store = FAISS.from_documents(chunks, embeddings)
print(f"✅ Vector store ready! {vector_store.index.ntotal} vectors")

# ─────────────────────────────────────────────
# STEP 6: CREATE RETRIEVER
# ─────────────────────────────────────────────
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
print("✅ Retriever ready!")

# ─────────────────────────────────────────────
# STEP 7: INITIALIZE LLM
# ─────────────────────────────────────────────
llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",
    temperature=0.2,
    max_tokens=512,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
print("✅ LLM ready!")

# ─────────────────────────────────────────────
# STEP 8: CREATE PROMPT TEMPLATE
# ─────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the following context:
    {context}
    
    Question: {question}
    """
)
print("✅ Prompt template ready!")

# ─────────────────────────────────────────────
# STEP 9: FORMAT DOCUMENTS
# ─────────────────────────────────────────────
def format_docs(retrieved_docs):
    """Format docs with source information."""
    parts = []
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        sheet = doc.metadata.get("sheet", "")
        
        label = f"[Doc {i+1} | {os.path.basename(source)}"
        if page: label += f" | Page {page}"
        if sheet: label += f" | Sheet: {sheet}"
        label += "]"
        
        parts.append(f"{label}\n{doc.page_content}")
    
    return "\n\n".join(parts)

# ─────────────────────────────────────────────
# STEP 10: BUILD RAG CHAIN
# ─────────────────────────────────────────────
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

main_chain = parallel_chain | prompt | llm | parser
print("✅ RAG chain ready!")

# ─────────────────────────────────────────────
# STEP 11: INTERACTIVE Q&A
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("🤖 Interactive RAG Q&A")
print("Type your question below. Type 'exit' to stop.")
print("="*60 + "\n")

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ["exit", "quit", "q"]:
        print("👋 Goodbye!")
        break
    
    if not user_input:
        continue
    
    print("⏳ Thinking...")
    response = main_chain.invoke(user_input)
    print(f"\n🤖 Answer: {response}\n")
    print("-" * 60)
```

---

## ⚙️ Configuration Options

### Tuning Parameters

#### Chunking (Step 4)
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Adjust based on your documents
    chunk_overlap=200   # Overlap for context preservation
)
```

**Recommendations:**
- Short documents, specific Q&A: `chunk_size=500, overlap=100`
- Long documents, broad Q&A: `chunk_size=2000, overlap=300`
- Default (balanced): `chunk_size=1000, overlap=200`

#### Retrieval (Step 6)
```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}  # Number of chunks to retrieve
)
```

**Tuning `k`:**
- `k=2`: Fast, minimal context (good for factual questions)
- `k=4`: Balanced (default, recommended)
- `k=8`: Comprehensive context (good for complex questions)

#### LLM Settings (Step 7)
```python
llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",
    temperature=0.2,    # 0.0=deterministic, 1.0=creative
    max_tokens=512,     # Max response length
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```

**Model choices:**
- `meta-llama/llama-3.3-70b-instruct`: Best quality (50 tokens = ~$0.00014)
- `gpt-3.5-turbo`: Fast, cheap
- `mistralai/mistral-7b-instruct-v0.1`: Lightweight

**Temperature tuning:**
- `0.0`: Deterministic (FAQ, factual Q&A)
- `0.5`: Balanced
- `1.0`: Creative (brainstorming, creative writing)

#### Embeddings (Step 5)
```python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  # Change model here
    model_kwargs={"device": "cpu"},        # "cuda" for GPU
)
```

**Embedding models:**
- `BAAI/bge-small-en-v1.5` (default): Fast, good quality
- `sentence-transformers/all-MiniLM-L6-v2`: Very lightweight
- `sentence-transformers/all-mpnet-base-v2`: Higher quality, slower

---

## 💾 Saving & Loading Vector Store

### Save to Disk (After building)
```python
# Save vector store for reuse
vector_store.save_local("faiss_index")
print("✅ Vector store saved to 'faiss_index/' folder")
```

### Load from Disk (Skip doc loading/chunking)
```python
# Load previously saved vector store
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
print("✅ Vector store loaded from disk!")
```

**Benefits:**
- Skip document loading/chunking/embedding (slow)
- Immediate Q&A on next run
- Useful for frequently-used documents

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'langchain_openai'"
**Solution:**
```bash
pip install langchain-openai
```

### Issue: "HuggingFace embedding model download fails"
**Solution:**
```bash
# Pre-download model explicitly
python -c "from langchain_community.embeddings import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')"
```

### Issue: "OpenRouter API: Invalid API key"
**Solution:**
1. Get new key from https://openrouter.ai/keys
2. Verify key format: starts with `sk-or-v1-`
3. Check for extra spaces: `key.strip()`
4. Verify environment variable set: `print(os.environ.get("OPENROUTER_API_KEY"))`

### Issue: "FAISS vector store empty or slow"
**Solution:**
- Reduce `chunk_size` (more, smaller chunks)
- Use GPU: `model_kwargs={"device": "cuda"}`
- Load from disk if already built

### Issue: "Out of Memory during embedding"
**Solution:**
```bash
# Use a smaller embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
# Or process documents in batches
```

### Issue: "PDF not loading - blank pages"
**Solution:**
- Some PDFs are image-based (scanned documents)
- Use OCR tool: `pytesseract` or `pdf2image`
- Manually extract text from problematic PDFs

---

## 📊 Performance Tuning Checklist

| Task | Optimization |
|------|-------------|
| **Slow embedding** | Use smaller model or GPU (`device: "cuda"`) |
| **Slow retrieval** | Use smaller `chunk_size` or smaller `k` |
| **Poor answer quality** | Increase `k`, improve chunks, use larger LLM |
| **High API cost** | Use cheaper model or increase `temperature` slightly |
| **Memory issues** | Reduce `chunk_size`, batch process, use disk caching |
| **Slow startup** | Load vector store from disk, not from scratch |

---

## 📚 Additional Resources

### OpenRouter API
- Website: https://openrouter.ai/
- API Keys: https://openrouter.ai/keys
- Model Pricing: https://openrouter.ai/models
- Documentation: https://openrouter.io/docs

### LangChain Documentation
- Main: https://python.langchain.com/
- RAG Template: https://python.langchain.com/docs/templates/rag_qdrant/
- FAISS: https://python.langchain.com/docs/integrations/vectorstores/faiss/

### HuggingFace
- Embedding Models: https://huggingface.co/sentence-transformers
- Model Cards: https://huggingface.co/models

### Further Learning
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)
- [Prompt Engineering](https://github.com/dair-ai/Prompt-Engineering-Guide)

---

## 📝 Version Info

```
Python: 3.8+
LangChain: 0.3.25+
FAISS: 1.13.2
HuggingFace Embeddings: Latest
OpenRouter API: v1
```

---

## 📄 Files in This Project

| File | Purpose |
|------|---------|
| `rag_agent_knowledge_sourrce.py` | Main Python script (runnable) |
| `RAG_for_Agent_Knowledge_Sourrce (4).ipynb` | Jupyter Notebook (interactive) |
| `README.md` | This documentation |
| `.env` | Environment variables (create this) |
| `venvt/` | Python virtual environment |
| `faiss_index/` | Saved vector store (optional) |

---

## 🤝 Quick Start Summary

1. **Install dependencies** → `pip install langchain langchain-openai faiss-cpu sentence-transformers...`
2. **Get API key** → Sign up at https://openrouter.ai/keys
3. **Configure file paths** → Update `FILE_PATHS` list
4. **Run script** → `python rag_agent_knowledge_sourrce.py`
5. **Ask questions** → Type in interactive Q&A loop
6. **Exit** → Type `exit` or `quit`

---

## 📞 Support & Questions

For issues or questions:
1. Check **Troubleshooting** section above
2. Review **LangChain Documentation**
3. Check OpenRouter API status
4. Verify API key and file paths

---

**Last Updated:** March 2026  
**Version:** 1.0
