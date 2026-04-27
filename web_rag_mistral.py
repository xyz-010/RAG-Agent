"""
Web Interface for RAG Chatbot with Mistral
"""

from flask import Flask, render_template, request, jsonify
from rag_chatbot_mistral import UniversityAdmissionsRAG
import os

app = Flask(__name__)

# Initialize chatbot (will be done on first request to avoid startup delay)
chatbot = None

def get_chatbot():
    """Lazy initialization of chatbot"""
    global chatbot
    if chatbot is None:
        print("Initializing RAG chatbot with Mistral...")
        chatbot = UniversityAdmissionsRAG()
    return chatbot

@app.route('/')
def home():
    """Render the main chat interface"""
    return render_template('rag_index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Get chatbot instance
        bot = get_chatbot()
        
        # Get response from RAG chatbot
        response = bot.ask(user_message)
        
        return jsonify({
            'response': response['answer'],
            'sources': response.get('sources', [])
        })
    except Exception as e:
        return jsonify({
            'error': f'Error: {str(e)}',
            'response': 'Sorry, I encountered an error. Please try again.'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'model': 'Mistral via Ollama'})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("\n" + "="*70)
    print("🎓 RAG Chatbot with Mistral - Web Interface")
    print("="*70)
    print("\n✅ Server starting...")
    print("📱 Open your browser to: http://127.0.0.1:5002")
    print("🤖 Using: Mistral LLM via Ollama")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    app.run(debug=False, host='127.0.0.1', port=5002, threaded=True)


