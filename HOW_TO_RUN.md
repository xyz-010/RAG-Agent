# How to Run - SPPU Admissions Chatbot

## Prerequisites

- Python 3.8 or higher
- Ollama installed with Mistral model
- Git (for cloning the repository)

## Installation Steps

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd "NLP project"
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and Setup Ollama

#### macOS/Linux:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Mistral model
ollama pull mistral
```

#### Windows:
- Download Ollama from https://ollama.com/download
- Install and run Ollama
- Open terminal and run: `ollama pull mistral`

### 5. Setup Environment Variables (Optional)
```bash
cp .env.example .env
# Edit .env if needed
```

### 6. Prepare Knowledge Base
Ensure the `knowledge_base/` directory contains your text files:
```
knowledge_base/
└── admissions_info.txt
```

## Running the Applications

### Option 1: Streamlit Web App (Recommended)
```bash
streamlit run streamlit_app.py
```
- Opens automatically in browser at http://localhost:8501
- Modern UI with dark mode
- Chat interface with sources

### Option 2: Flask Web App with RAG
```bash
python3 web_rag_mistral.py
```
- Access at http://localhost:5000
- Simple web interface
- RAG-powered responses

### Option 3: Simple Chatbot (Terminal)
```bash
python3 simple_chatbot.py
```
- Command-line interface
- Quick testing
- No RAG, direct Mistral responses

### Option 4: RAG Chatbot (Terminal)
```bash
python3 rag_chatbot_mistral.py
```
- Command-line interface with RAG
- Shows source documents
- Best for testing RAG functionality

## Troubleshooting

### Ollama Not Running
```bash
# Check if Ollama is running
ollama list

# Start Ollama service (if needed)
ollama serve
```

### Port Already in Use
```bash
# Kill existing Streamlit processes
pkill -9 -f streamlit

# Or use a different port
streamlit run streamlit_app.py --server.port 8502
```

### ChromaDB Issues
```bash
# Delete and recreate vector database
rm -rf chroma_db/
# Run the app again to recreate
```

### Missing Dependencies
```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt
```

## Project Structure

```
NLP project/
├── streamlit_app.py          # Main Streamlit app
├── web_rag_mistral.py         # Flask web app
├── simple_chatbot.py          # Simple terminal chatbot
├── rag_chatbot_mistral.py     # RAG terminal chatbot
├── requirements.txt           # Python dependencies
├── setup.sh                   # Setup script
├── knowledge_base/            # Text files for RAG
│   └── admissions_info.txt
├── templates/                 # HTML templates
│   ├── index.html
│   └── rag_index.html
├── chroma_db/                 # Vector database (auto-generated)
└── venv/                      # Virtual environment

```

## Features

### Streamlit App
- ✅ Modern dark mode UI
- ✅ Chat history
- ✅ Source citations
- ✅ Recommended questions
- ✅ Real-time responses

### RAG System
- ✅ Document retrieval
- ✅ Context-aware answers
- ✅ Source tracking
- ✅ Mistral AI integration

## Testing

### Test the Chatbot
```bash
python3 test_chatbot.py
```

### Sample Questions
- "What are the admission requirements for B.Tech?"
- "How do I apply through MHT-CET?"
- "What is the fee structure?"
- "Tell me about hostel facilities"

## Stopping the Applications

### Streamlit
- Press `Ctrl+C` in the terminal
- Or: `pkill -9 -f streamlit`

### Flask
- Press `Ctrl+C` in the terminal

## Additional Notes

- First run will take time to create vector database
- Subsequent runs will be faster (uses cached embeddings)
- Ensure Ollama is running before starting any app
- Knowledge base can be updated by adding/modifying files in `knowledge_base/`

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify Ollama is running: `ollama list`
3. Check Python version: `python3 --version`
4. Ensure all dependencies are installed: `pip list`

## Quick Start (TL;DR)

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull mistral

# Run
streamlit run streamlit_app.py
```

