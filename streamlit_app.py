import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

# Page configuration
st.set_page_config(
    page_title="SPPU Admissions Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Gemini-style Dark Mode CSS
st.markdown("""
<style>
    /* Main background - Dark */
    .main {
        background-color: #1e1e1e;
        color: #e8eaed;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Streamlit elements dark mode */
    .stApp {
        background-color: #1e1e1e;
    }
    
    /* Header styling - Dark */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .header-title {
        color: white;
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* Message styling - Dark Gemini style */
    .message {
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-radius: 16px;
        line-height: 1.6;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .assistant-message {
        background: #2d2d2d;
        color: #e8eaed;
        margin-right: 20%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border: 1px solid #3c4043;
    }
    
    .message-role {
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .user-message .message-role {
        color: rgba(255,255,255,0.9);
    }
    
    .assistant-message .message-role {
        color: #9aa0a6;
    }
    
    .message-content {
        font-size: 1rem;
        white-space: pre-wrap;
    }
    
    /* Source styling - Dark */
    .source-container {
        margin-top: 1rem;
        padding: 1rem;
        background: #2d2d2d;
        border-radius: 8px;
        border-left: 3px solid #667eea;
    }
    
    .source-title {
        font-weight: 600;
        color: #9aa0a6;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    
    .source-item {
        font-size: 0.85rem;
        color: #9aa0a6;
        padding: 0.5rem;
        background: #1e1e1e;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling - Dark */
    .sidebar .sidebar-content {
        background: #2d2d2d;
    }
    
    /* Sidebar text color */
    section[data-testid="stSidebar"] {
        background-color: #2d2d2d;
        color: #e8eaed;
    }
    
    section[data-testid="stSidebar"] * {
        color: #e8eaed !important;
    }
    
    /* Suggestion chips */
    .suggestion-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .chip {
        background: white;
        border: 1px solid #dadce0;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
        color: #1f1f1f;
        cursor: pointer;
        transition: all 0.2s;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .chip:hover {
        background: #f8f9fa;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }
    
    /* Welcome screen - Dark */
    .welcome-container {
        text-align: center;
        padding: 3rem 1rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .welcome-title {
        font-size: 3rem;
        font-weight: 400;
        color: #e8eaed;
        margin-bottom: 1rem;
    }
    
    .welcome-subtitle {
        font-size: 1.2rem;
        color: #9aa0a6;
        margin-bottom: 2rem;
    }
    
    /* Loading animation */
    .loading {
        display: flex;
        gap: 0.5rem;
        padding: 1rem;
    }
    
    .loading-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #667eea;
        animation: bounce 1.4s infinite ease-in-out both;
    }
    
    .loading-dot:nth-child(1) { animation-delay: -0.32s; }
    .loading-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    /* Button styling - Dark */
    .stButton > button {
        border-radius: 20px;
        border: 1px solid #3c4043;
        background: #2d2d2d;
        color: #e8eaed;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #3c4043;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }
    
    /* Input styling - Dark */
    .stChatInput > div {
        background-color: #2d2d2d;
        border: 1px solid #3c4043;
    }
    
    .stChatInput input {
        background-color: #2d2d2d;
        color: #e8eaed;
    }
    
    /* Expander styling - Dark */
    .streamlit-expanderHeader {
        background-color: #2d2d2d;
        color: #e8eaed;
    }
    
    /* Markdown text - Dark */
    .stMarkdown {
        color: #e8eaed;
    }
    
    /* Headings - Dark */
    h1, h2, h3, h4, h5, h6 {
        color: #e8eaed !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None

if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# RAG Chatbot Class
class SPPUAdmissionsChatbot:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        
    def initialize(self):
        """Initialize the RAG pipeline"""
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Check if vector store exists
        persist_directory = "./chroma_db"
        
        if os.path.exists(persist_directory):
            # Load existing vector store
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
        else:
            # Create new vector store
            loader = DirectoryLoader(
                'knowledge_base/',
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents = loader.load()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            texts = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            self.vectorstore.persist()
        
        # Initialize Mistral LLM
        self.llm = Ollama(
            model="mistral",
            temperature=0.7,
            top_p=0.9
        )
        
        # Create custom prompt
        prompt_template = """You are a helpful assistant for Savitribai Phule Pune University (SPPU) admissions.
Use the following context to answer the question. Provide detailed, accurate information.
If you don't know the answer, say so honestly.

Context: {context}

Question: {question}

Answer: """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def ask(self, question):
        """Ask a question and get response"""
        if not self.qa_chain:
            return {"answer": "Chatbot not initialized. Please wait...", "sources": []}
        
        try:
            response = self.qa_chain({"query": question})
            
            # Extract source information
            sources = []
            if 'source_documents' in response:
                for doc in response['source_documents']:
                    source_info = {
                        'content': doc.page_content[:150] + "...",
                        'source': doc.metadata.get('source', 'Unknown')
                    }
                    sources.append(source_info)
            
            return {
                "answer": response['result'],
                "sources": sources
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "sources": []
            }

# Initialize chatbot
@st.cache_resource
def get_chatbot():
    with st.spinner("🔄 Initializing SPPU Assistant..."):
        chatbot = SPPUAdmissionsChatbot()
        chatbot.initialize()
    return chatbot

# Sidebar with recommended questions
with st.sidebar:
    st.markdown("### Recommended Questions")
    st.markdown("Click any question to ask:")
    
    questions = [
        "What are B.Tech admission requirements?",
        "How do I apply through MHT-CET?",
        "What is the fee structure?",
        "What scholarships are available?",
        "Tell me about hostel facilities",
        "What is the placement scenario?",
        "What programs are offered?",
        "What are the admission deadlines?",
        "How do I reach SPPU campus?",
        "What are the contact details?"
    ]
    
    for question in questions:
        if st.button(question, key=question, use_container_width=True):
            st.session_state.current_question = question
            st.rerun()
    
    st.markdown("---")
    
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### About
    **University**: SPPU, Pune
    **Technology**: RAG + Mistral AI
    **Status**: Online
    """)

# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">
        SPPU Admissions Assistant
    </div>
    <div class="header-subtitle">
        Powered by AI • Savitribai Phule Pune University
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize chatbot
if not st.session_state.initialized:
    try:
        st.session_state.chatbot = get_chatbot()
        st.session_state.initialized = True
    except Exception as e:
        st.error(f"❌ Error initializing chatbot: {str(e)}")
        st.stop()

# Welcome screen or chat
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-title">Hello! How can I help you today?</div>
        <div class="welcome-subtitle">Ask me anything about SPPU admissions, programs, fees, or campus life</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Suggestion chips
    st.markdown("### Quick Start")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Admissions", use_container_width=True):
            st.session_state.current_question = "What are the admission requirements for B.Tech?"
            st.rerun()
    
    with col2:
        if st.button("Fees", use_container_width=True):
            st.session_state.current_question = "What is the fee structure for engineering courses?"
            st.rerun()
    
    with col3:
        if st.button("MHT-CET", use_container_width=True):
            st.session_state.current_question = "How do I apply through MHT-CET?"
            st.rerun()

else:
    # Display chat messages
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="message user-message">
                <div class="message-role">You</div>
                <div class="message-content">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message assistant-message">
                <div class="message-role">Assistant</div>
                <div class="message-content">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Always display sources section
            if message.get("sources"):
                st.markdown("### Sources")
                for i, source in enumerate(message["sources"], 1):
                    with st.expander(f"Source {i}: {source['source']}", expanded=False):
                        st.markdown(f"""
                        <div class="source-item">
                            {source['content']}
                        </div>
                        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input
if 'current_question' in st.session_state:
    user_input = st.session_state.current_question
    del st.session_state.current_question
else:
    user_input = st.chat_input("Ask me anything about SPPU admissions...")

if user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Show loading animation
    with st.spinner(""):
        st.markdown("""
        <div class="loading">
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Get response
        if st.session_state.chatbot is None:
            st.error("❌ Chatbot not initialized. Please refresh the page.")
            st.stop()
        
        response = st.session_state.chatbot.ask(user_input)
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response.get("sources", [])
        })
    
    # Rerun to display new messages
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #9aa0a6; padding: 1rem; font-size: 0.9rem;'>
    <p>Powered by RAG Technology + Mistral AI</p>
    <p>For official information, visit <a href='https://www.unipune.ac.in' target='_blank' style='color: #8ab4f8; text-decoration: none;'>www.unipune.ac.in</a></p>
</div>
""", unsafe_allow_html=True)

