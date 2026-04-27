# Architecture Documentation

## System Overview

The University Admissions Chatbot uses **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware responses to user queries about university admissions.

## RAG Architecture

### What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that combines:
1. **Information Retrieval**: Finding relevant documents from a knowledge base
2. **Text Generation**: Using an LLM to generate natural language responses

This approach ensures responses are:
- Grounded in factual information
- Up-to-date (by updating the knowledge base)
- Traceable (can show sources)

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│            (Streamlit / Flask Web UI / CLI)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG Chatbot Engine                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Question Processing                               │  │
│  │     - Text normalization                              │  │
│  │     - Intent detection                                │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  2. Embedding Generation                              │  │
│  │     - Convert question to vector                      │  │
│  │     - Model: sentence-transformers/all-MiniLM-L6-v2   │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  3. Document Retrieval                                │  │
│  │     - Semantic similarity search                      │  │
│  │     - Retrieve top-k relevant chunks                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  4. Context Assembly                                  │  │
│  │     - Combine retrieved documents                     │  │
│  │     - Format for LLM input                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  5. Response Generation                               │  │
│  │     - LLM generates answer from context               │  │
│  │     - Model: Mistral (via Ollama)                     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Vector Database                           │
│                      (ChromaDB)                              │
│  - Stores document embeddings                                │
│  - Enables fast similarity search                            │
│  - Persists to disk                                          │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Knowledge Base                            │
│                  (Text Documents)                            │
│  - admissions_info.txt                                       │
│  - Additional documents can be added                         │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Initialization Phase

```python
# Load documents
documents = load_documents("knowledge_base/")

# Split into chunks
chunks = split_documents(documents, chunk_size=500)

# Generate embeddings
embeddings = embedding_model.encode(chunks)

# Store in vector database
vectorstore.add(chunks, embeddings)
```

### 2. Query Phase

```python
# User asks question
question = "What are the admission requirements?"

# Generate question embedding
question_embedding = embedding_model.encode(question)

# Retrieve similar documents
relevant_docs = vectorstore.similarity_search(
    question_embedding, 
    k=3
)

# Generate response
context = combine(relevant_docs)
prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
answer = llm.generate(prompt)
```

## Key Technologies

### 1. LangChain
- **Purpose**: Orchestrates the RAG pipeline
- **Components Used**:
  - `TextLoader`: Load documents
  - `RecursiveCharacterTextSplitter`: Split documents
  - `RetrievalQA`: QA chain with retrieval
  - `PromptTemplate`: Custom prompts

### 2. Sentence Transformers
- **Model**: `all-MiniLM-L6-v2`
- **Purpose**: Generate semantic embeddings
- **Dimensions**: 384
- **Speed**: ~1000 sentences/second on CPU

### 3. ChromaDB
- **Purpose**: Vector database for similarity search
- **Features**:
  - Persistent storage
  - Fast similarity search
  - Metadata filtering
  - Easy to use API

### 4. Mistral (via Ollama)
- **Model**: `mistral`
- **Purpose**: Generate natural language responses
- **Parameters**: 7B
- **Strengths**:
  - Excellent instruction following
  - Fast inference
  - Runs locally via Ollama
  - High quality responses

### 5. Ollama
- **Purpose**: Local LLM runtime
- **Features**:
  - Easy model management
  - Fast inference
  - No API keys needed
  - Privacy-focused (runs locally)

### 6. Streamlit
- **Purpose**: Primary web interface
- **Features**:
  - Modern, responsive UI
  - Real-time chat interface
  - Dark mode support
  - Easy deployment

### 7. Flask (Alternative)
- **Purpose**: Alternative web interface
- **Features**:
  - Lightweight
  - RESTful API
  - Custom HTML templates

## Document Processing

### Chunking Strategy

Documents are split into chunks for better retrieval:

```python
chunk_size = 500        # Characters per chunk
chunk_overlap = 50      # Overlap between chunks
```

**Why chunking?**
- Improves retrieval precision
- Fits within model context limits
- Balances specificity vs context

### Embedding Strategy

Each chunk is converted to a 384-dimensional vector:

```python
embedding = model.encode(chunk)
# embedding.shape = (384,)
```

**Semantic similarity** is computed using cosine similarity:

```python
similarity = cosine_similarity(query_embedding, doc_embedding)
```

## Retrieval Strategy

### Top-K Retrieval

Retrieve the top 3 most relevant chunks:

```python
k = 3  # Number of chunks to retrieve
results = vectorstore.similarity_search(query, k=k)
```

**Trade-offs:**
- Higher k: More context, but slower and may include irrelevant info
- Lower k: Faster, but may miss important context

### Similarity Threshold

Only retrieve documents above a similarity threshold:

```python
threshold = 0.7
results = [doc for doc in results if doc.score > threshold]
```

## Prompt Engineering

### Custom Prompt Template

```python
template = """You are a helpful university admissions assistant. 
Use the following context to answer the question.
If you don't know the answer, say so.

Context: {context}

Question: {question}

Answer: Let me help you with that."""
```

**Key elements:**
- Role definition (admissions assistant)
- Instruction to use context
- Fallback behavior
- Friendly tone

## Performance Optimization

### 1. Model Selection

**Embedding Model:**
- `all-MiniLM-L6-v2`: Fast, good quality (chosen)
- `all-mpnet-base-v2`: Better quality, slower
- `all-distilroberta-v1`: Balanced

**LLM:**
- `mistral` (via Ollama): Excellent quality, fast, local (chosen)
- `llama2` (via Ollama): Good alternative
- `mixtral` (via Ollama): Better quality, requires more resources
- OpenAI GPT: Best quality, requires API key and internet

### 2. Caching

- Vector database persisted to disk
- Models cached after first download
- Embeddings computed once

### 3. Batch Processing

Process multiple queries efficiently:

```python
embeddings = model.encode(queries, batch_size=32)
```

## Scalability Considerations

### Current Limitations

- Single-threaded processing
- CPU-only inference
- Limited to local models
- No distributed storage

### Scaling Options

1. **GPU Acceleration**
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   ```

2. **Model Quantization**
   - Reduce model size
   - Faster inference
   - Slight quality trade-off

3. **Distributed Vector Store**
   - Use Pinecone, Weaviate, or Milvus
   - Handle millions of documents
   - Cloud-based

4. **API-based LLMs**
   - OpenAI GPT-4
   - Anthropic Claude
   - Better quality, pay per use

## Security Considerations

### Data Privacy

- All processing done locally
- No data sent to external services (unless using API-based LLMs)
- Knowledge base stored locally

### Input Validation

```python
# Sanitize user input
question = question.strip()
if len(question) > 1000:
    question = question[:1000]
```

### Rate Limiting

For production deployment:

```python
from flask_limiter import Limiter
limiter = Limiter(app, default_limits=["100 per hour"])
```

## Monitoring and Logging

### Metrics to Track

1. **Response Time**
   - Embedding generation time
   - Retrieval time
   - LLM generation time

2. **Quality Metrics**
   - User satisfaction ratings
   - Answer relevance scores
   - Source attribution accuracy

3. **Usage Metrics**
   - Questions per day
   - Popular topics
   - Unanswered questions

### Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Question: {question}")
logger.info(f"Retrieved {len(docs)} documents")
logger.info(f"Response time: {elapsed_time}s")
```

## Future Enhancements

### 1. Multi-Modal Support
- Image understanding (campus photos)
- PDF document upload
- Voice input/output

### 2. Personalization
- User profiles
- Conversation history
- Personalized recommendations

### 3. Advanced RAG Techniques
- **Hybrid Search**: Combine semantic + keyword search
- **Re-ranking**: Re-order retrieved documents
- **Query Expansion**: Generate multiple query variations
- **Self-RAG**: Model evaluates its own responses

### 4. Integration
- University database integration
- Calendar integration
- Email notifications
- CRM integration

## References

- [LangChain Documentation](https://python.langchain.com/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)