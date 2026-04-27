#!/bin/bash

# Setup script for SPPU Admissions Chatbot with RAG + Mistral

echo "=========================================="
echo "SPPU Admissions Chatbot Setup"
echo "RAG + Mistral AI"
echo "=========================================="
echo ""

# Check if Ollama is installed
echo "Checking for Ollama..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
    ollama_version=$(ollama --version 2>&1)
    echo "  Version: $ollama_version"
else
    echo "✗ Ollama is not installed"
    echo ""
    echo "Please install Ollama first:"
    echo "  macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Windows: Download from https://ollama.com/download"
    echo ""
    exit 1
fi
echo ""

# Check if Mistral model is available
echo "Checking for Mistral model..."
if ollama list | grep -q "mistral"; then
    echo "✓ Mistral model is available"
else
    echo "✗ Mistral model not found"
    echo ""
    echo "Pulling Mistral model (this will take a few minutes)..."
    ollama pull mistral
    if [ $? -eq 0 ]; then
        echo "✓ Mistral model downloaded successfully"
    else
        echo "✗ Failed to download Mistral model"
        exit 1
    fi
fi
echo ""

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✓ Found Python $python_version"
    
    # Check if version is 3.8 or higher
    major=$(echo $python_version | cut -d. -f1)
    minor=$(echo $python_version | cut -d. -f2)
    if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then
        echo "✓ Python version is compatible"
    else
        echo "✗ Python 3.8 or higher is required"
        exit 1
    fi
else
    echo "✗ Python 3 is not installed"
    exit 1
fi
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "✓ Virtual environment already exists"
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies (this may take 10-15 minutes)..."
echo "This includes: LangChain, ChromaDB, Streamlit, Flask, and more..."
pip install -r requirements.txt --quiet
if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi
echo ""

# Download embedding model
echo "Downloading embedding model (sentence-transformers)..."
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Embedding model downloaded"
else
    echo "✗ Failed to download embedding model"
fi
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✓ .env file created from template"
    else
        echo "✓ .env.example not found, skipping"
    fi
else
    echo "✓ .env file already exists"
fi
echo ""

# Check if knowledge base exists
if [ -d "knowledge_base" ] && [ -f "knowledge_base/admissions_info.txt" ]; then
    echo "✓ Knowledge base found"
else
    echo "⚠ Warning: knowledge_base/admissions_info.txt not found"
    echo "  The chatbot needs this file to work properly"
fi
echo ""

echo "=========================================="
echo "Setup Complete! 🎉"
echo "=========================================="
echo ""
echo "To run the chatbot, choose one of these options:"
echo ""
echo "1. Streamlit Web App (Recommended):"
echo "   source venv/bin/activate"
echo "   streamlit run streamlit_app.py"
echo "   Then open: http://localhost:8501"
echo ""
echo "2. Flask Web App:"
echo "   source venv/bin/activate"
echo "   python3 web_rag_mistral.py"
echo "   Then open: http://localhost:5002"
echo ""
echo "3. CLI with RAG:"
echo "   source venv/bin/activate"
echo "   python3 rag_chatbot_mistral.py"
echo ""
echo "4. Simple CLI:"
echo "   source venv/bin/activate"
echo "   python3 simple_chatbot.py"
echo ""
echo "Note: First run will take 30-60 seconds to build the vector database."
echo ""
echo "For detailed instructions, see HOW_TO_RUN.md"
echo ""


