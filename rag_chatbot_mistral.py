"""
RAG-based University Admissions Chatbot using Mistral via Ollama
"""

import os
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader, DirectoryLoader


class UniversityAdmissionsRAG:
    """RAG-based chatbot using Mistral via Ollama"""
    
    def __init__(self, knowledge_base_path: str = "knowledge_base", 
                 persist_directory: str = "chroma_db"):
        """
        Initialize the RAG chatbot with Mistral
        
        Args:
            knowledge_base_path: Path to directory containing knowledge base documents
            persist_directory: Path to persist vector database
        """
        self.knowledge_base_path = knowledge_base_path
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.qa_chain = None
        self.conversation_history = []
        
        print("=" * 70)
        print("🎓 University Admissions Chatbot (RAG + Mistral)")
        print("=" * 70)
        print("\nInitializing chatbot...")
        self._setup_embeddings()
        self._load_and_process_documents()
        self._setup_llm()
        self._create_qa_chain()
        print("\n✅ Chatbot ready!")
        print("=" * 70)
    
    def _setup_embeddings(self):
        """Setup embedding model for document vectorization"""
        print("📊 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✓ Embedding model loaded")
    
    def _load_and_process_documents(self):
        """Load documents from knowledge base and create vector store"""
        print("📚 Loading knowledge base documents...")
        
        # Check if vector store already exists
        if os.path.exists(self.persist_directory):
            print("✓ Loading existing vector database...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("✓ Vector database loaded")
            return
        
        # Check if knowledge base path exists
        if not os.path.exists(self.knowledge_base_path):
            raise FileNotFoundError(
                f"Knowledge base directory not found: {self.knowledge_base_path}"
            )
        
        # Load documents
        loader = DirectoryLoader(
            self.knowledge_base_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        
        if not documents:
            raise ValueError(
                f"No documents found in {self.knowledge_base_path}. "
                "Please add .txt files to the knowledge base directory."
            )
        
        print(f"✓ Loaded {len(documents)} documents")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_documents(documents)
        print(f"✓ Split into {len(texts)} chunks")
        
        # Create vector store
        print("🔄 Creating vector database (this may take a minute)...")
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()
        print("✓ Vector database created and persisted")
    
    def _setup_llm(self):
        """Setup Mistral LLM via Ollama"""
        print("🤖 Connecting to Mistral via Ollama...")
        
        self.llm = Ollama(
            model="mistral",
            temperature=0.7,
            top_p=0.9,
        )
        print("✓ Mistral LLM connected")
    
    def _create_qa_chain(self):
        """Create the QA chain with custom prompt"""
        
        # Ensure vectorstore is initialized
        if self.vectorstore is None:
            raise RuntimeError(
                "Vector store not initialized. Cannot create QA chain."
            )
        
        # Custom prompt template optimized for Mistral
        template = """You are a helpful and friendly university admissions assistant. Use the following context to answer the question accurately and concisely.

Context: {context}

Question: {question}

Answer: """
        
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Retrieve top 3 relevant chunks
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        print("✓ RAG pipeline configured")
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question to the chatbot
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and source information
        """
        # Handle greetings
        greetings = ["hi", "hello", "hey", "greetings"]
        if question.lower().strip() in greetings:
            return {
                "answer": "Hello! I'm the University Admissions Assistant powered by Mistral AI. I can help you with questions about admission requirements, deadlines, programs, financial aid, campus life, and more. What would you like to know?",
                "sources": []
            }
        
        # Handle goodbyes
        goodbyes = ["bye", "goodbye", "thanks", "thank you"]
        if any(word in question.lower() for word in goodbyes):
            return {
                "answer": "You're welcome! If you have more questions, feel free to ask anytime. You can also contact the admissions office at admissions@university.edu or call (555) 123-4567. Good luck with your application!",
                "sources": []
            }
        
        # Ensure QA chain is initialized
        if self.qa_chain is None:
            return {
                "answer": "I'm sorry, but the chatbot is not properly initialized. Please restart the application.",
                "sources": []
            }
        
        # Process question through RAG pipeline
        try:
            print("\n🔍 Searching knowledge base...")
            result = self.qa_chain({"query": question})
            
            # Extract answer and sources
            answer = result["result"]
            sources = result.get("source_documents", [])
            
            # Add to conversation history
            self.conversation_history.append({
                "question": question,
                "answer": answer
            })
            
            return {
                "answer": answer,
                "sources": [doc.page_content[:200] + "..." for doc in sources[:2]]
            }
        except Exception as e:
            return {
                "answer": f"I encountered an error processing your question: {str(e)}. Please try rephrasing it or contact the admissions office directly.",
                "sources": []
            }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


def main():
    """Main function to run the chatbot in CLI mode"""
    
    # Initialize chatbot
    chatbot = UniversityAdmissionsRAG()
    
    print("\n💬 You can now ask questions! Type 'quit' or 'exit' to end.\n")
    
    while True:
        # Get user input
        question = input("You: ").strip()
        
        if not question:
            continue
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using the University Admissions Chatbot!")
            break
        
        # Get response
        response = chatbot.ask(question)
        
        # Display answer
        print(f"\n🤖 Bot: {response['answer']}\n")
        
        # Optionally display sources
        if response['sources']:
            print("📚 Sources:")
            for i, source in enumerate(response['sources'], 1):
                print(f"  {i}. {source}")
            print()


if __name__ == "__main__":
    main()


