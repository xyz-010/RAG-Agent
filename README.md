# SPPU Admissions Chatbot - RAG with Mistral AI

An intelligent chatbot for answering questions about **Savitribai Phule Pune University (SPPU)** admissions using Retrieval-Augmented Generation (RAG) technology powered by Mistral AI.

## 🎓 About

This chatbot provides comprehensive information about SPPU admissions including:
- Undergraduate and Postgraduate programs
- MHT-CET and entrance exam details
- CAP (Centralized Admission Process)
- Fee structure and scholarships
- Reservation policies (SC/ST/OBC/EWS/PwD)
- Campus facilities and placements
- And much more!

## 🚀 Features

- **RAG Technology**: Retrieves relevant information from knowledge base before generating responses
- **Mistral AI**: Uses Mistral 7B LLM running locally via Ollama for natural language generation
- **Vector Search**: ChromaDB for efficient semantic search
- **Web Interface**: Beautiful, responsive web UI
- **Source Attribution**: Shows which documents were used to generate answers
- **Real-time Responses**: Fast and accurate answers

## 📋 Prerequisites

- Python 3.9+
- Ollama installed with Mistral model
- 8GB+ RAM recommended
- macOS, Linux, or Windows

## 🛠️ Installation

### 1. Install Ollama and Mistral

**macOS/Linux:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Or on macOS with Homebrew
brew install ollama

# Pull Mistral model (4.4 GB)
ollama pull mistral
```

**Windows:**
1. Download Ollama from https://ollama.com/download
2. Install and run Ollama
3. Open terminal and run: `ollama pull mistral`

### 2. Clone the Repository

```bash
# Clone the repository
git clone <your-repository-url>
cd <repository-name>
```

### 3. Setup Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Check if Ollama is running
ollama list

# Should show mistral in the list
```

## 🎯 Usage

### Option 1: Streamlit Web App (Recommended)

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Run Streamlit app
streamlit run streamlit_app.py
```

Then open your browser to: **http://localhost:8501**

**Features:**
- Modern, responsive UI with dark mode
- Real-time chat interface
- Source citations
- Recommended questions
- Chat history

### Option 2: Flask Web App

```bash
# Activate virtual environment
source venv/bin/activate

# Run Flask server
python3 web_rag_mistral.py
```

Then open your browser to: **http://127.0.0.1:5002**

### Option 3: Command Line Interface

**RAG Chatbot (with knowledge base):**
```bash
source venv/bin/activate
python3 rag_chatbot_mistral.py
```

**Simple Chatbot (direct Mistral):**
```bash
source venv/bin/activate
python3 simple_chatbot.py
```

## 📚 Knowledge Base

The chatbot uses a comprehensive knowledge base covering:

- **Admissions**: Requirements, deadlines, application process
- **Entrance Exams**: MHT-CET, GATE, MAH-CET details
- **Programs**: B.A./B.Sc./B.Com, B.Tech/B.E., MBA, MCA, Ph.D.
- **Fees**: Detailed fee structure for all programs
- **Scholarships**: Government and merit-based scholarships
- **Reservations**: SC/ST/OBC/EWS/PwD quotas
- **Campus**: Hostels, facilities, sports, cultural activities
- **Placements**: Statistics, top recruiters, salary packages
- **Location**: Campus address, connectivity, transport

## 💡 Example Questions

Try asking:
- "What are the admission requirements for B.Tech?"
- "How do I apply through MHT-CET?"
- "What is the CAP process?"
- "Tell me about OBC reservation policy"
- "What scholarships are available?"
- "What is the fee structure for engineering?"
- "Which companies come for placements?"
- "Tell me about hostel facilities"
- "How do I reach SPPU campus?"

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Web Interface  │ (Flask)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   RAG Pipeline  │ (LangChain)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│ChromaDB│ │Mistral │
│Vector  │ │  LLM   │
│  DB    │ │(Ollama)│
└────────┘ └────────┘
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│    Response     │
└─────────────────┘
```

## 📁 Project Structure

```
project/
├── streamlit_app.py           # Streamlit web app (main)
├── web_rag_mistral.py         # Flask web server
├── rag_chatbot_mistral.py     # CLI chatbot with RAG
├── simple_chatbot.py          # Simple CLI chatbot
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── HOW_TO_RUN.md             # Detailed setup guide
├── ARCHITECTURE.md           # System architecture docs
├── .gitignore                # Git ignore rules
├── .env.example              # Environment variables template
├── knowledge_base/
│   └── admissions_info.txt   # SPPU admissions information
├── templates/
│   ├── index.html            # Simple web UI
│   └── rag_index.html        # Flask RAG UI
└── chroma_db/                # Vector database (auto-generated)
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file (optional):

```bash
OLLAMA_MODEL=mistral
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### Customization

To update the knowledge base:
1. Edit `knowledge_base/admissions_info.txt`
2. Delete `chroma_db/` folder
3. Restart the chatbot (it will rebuild the vector database)

## 🐛 Troubleshooting

### Port Already in Use

If port 5002 is busy:
```bash
# Kill existing process
pkill -f web_rag_mistral.py

# Or change port in web_rag_mistral.py
app.run(host='0.0.0.0', port=5003)
```

### Mistral Not Found

```bash
# Check if Ollama is running
ollama list

# Pull Mistral if not installed
ollama pull mistral
```

### ChromaDB Issues

```bash
# Delete and rebuild vector database
rm -rf chroma_db/
python3 web_rag_mistral.py
```

### Slow First Response

The first query takes 30-60 seconds to build the vector database. Subsequent queries are fast.

## 📊 Performance

- **Average Response Time**: 2-5 seconds
- **First Query**: 30-60 seconds (building vector DB)
- **Accuracy**: High (RAG retrieves relevant context)
- **Memory Usage**: ~2-3 GB
- **Disk Space**: ~5 GB (including Mistral model)

## 🔒 Security

- Runs completely locally (no data sent to external servers)
- No API keys required
- Privacy-focused design

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

To add more information to the knowledge base:
1. Edit `knowledge_base/admissions_info.txt`
2. Add relevant SPPU information
3. Delete `chroma_db/` folder
4. Restart the chatbot

## 📧 Contact

For questions about SPPU admissions, visit: www.unipune.ac.in

## 🙏 Acknowledgments

- **SPPU** for admissions information
- **Mistral AI** for the language model
- **LangChain** for RAG framework
- **ChromaDB** for vector storage
- **Ollama** for local LLM deployment

## 🎯 Future Enhancements

- [ ] Add conversation history
- [ ] Multi-language support (Marathi, Hindi)
- [ ] Voice input/output
- [ ] Mobile app
- [ ] Integration with SPPU official website
- [ ] Real-time admission updates
- [ ] Chatbot analytics dashboard

---

