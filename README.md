# RAG AI Chatbot 🤖

A conversational AI chatbot built with Streamlit that uses Retrieval-Augmented Generation (RAG) to answer questions based on uploaded documents.

## Features ✨

- 📄 **Document Upload**: Support for PDF, DOCX, and TXT files
- 🤖 **Google Gemini**: Advanced LLM for response generation
- 🔤 **Ollama Embeddings**: Using nomic-embed-text model for high-quality embeddings
- 🗄️ **FAISS Vector Store**: Efficient similarity search for relevant context retrieval
- 💬 **Conversational Interface**: User-friendly Streamlit UI
- 📚 **Context-Aware Responses**: Answers based on your uploaded documents

## Prerequisites 📋

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running locally
3. **Google Gemini API Key** (get it from [Google AI Studio](https://makersuite.google.com/app/apikey))

## Installation Steps 🚀

### 1. Install Ollama

**For Linux/Mac:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**For Windows:**
Download from [ollama.com](https://ollama.com/download)

### 2. Pull the nomic-embed-text model

```bash
ollama pull nomic-embed-text
```

Verify Ollama is running:
```bash
ollama list
```

### 3. Clone/Download the project

```bash
# If using git
git clone <repository-url>
cd rag-chatbot

# Or download and extract the files
```

### 4. Create a virtual environment (recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 5. Install Python dependencies

```bash
pip install -r requirements.txt
```

## Usage 🎯

### 1. Start Ollama (if not already running)

```bash
ollama serve
```

Keep this terminal window open.

### 2. Run the Streamlit app

Open a new terminal window and run:

```bash
streamlit run rag_chatbot.py
```

The app will open in your default browser at `http://localhost:8501`

### 3. Configure the chatbot

1. **Enter your Google Gemini API Key** in the sidebar
2. **Upload documents** (optional):
   - Click "Browse files" 
   - Select PDF, DOCX, or TXT files
   - Click "Process Documents"
3. **Start chatting!** Type your questions in the chat input

## How It Works 🔧

1. **Document Ingestion**: Uploaded documents are parsed and converted to text
2. **Text Chunking**: Documents are split into manageable chunks (1000 chars with 200 overlap)
3. **Embedding Generation**: Ollama's nomic-embed-text model creates vector embeddings
4. **Vector Storage**: FAISS stores embeddings for fast similarity search
5. **Query Processing**: User questions are embedded and matched with relevant chunks
6. **Response Generation**: Google Gemini generates answers using retrieved context

## Architecture 🏗️

```
User Query
    ↓
Query Embedding (Ollama)
    ↓
Similarity Search (FAISS)
    ↓
Retrieve Relevant Context
    ↓
Generate Response (Google Gemini)
    ↓
Display to User
```

## Supported File Types 📁

- **PDF** (.pdf)
- **Word Documents** (.docx)
- **Text Files** (.txt)

## Troubleshooting 🔍

### Ollama Connection Error

**Error**: "Error creating vector store" or "Connection refused"

**Solution**: 
- Make sure Ollama is running: `ollama serve`
- Check if nomic-embed-text is installed: `ollama list`
- If not installed: `ollama pull nomic-embed-text`

### Google Gemini API Error

**Error**: "Error generating response"

**Solution**:
- Verify your API key is correct
- Check your API quota at [Google AI Studio](https://makersuite.google.com/)
- Ensure you have billing enabled if required

### Document Processing Issues

**Error**: "Error reading PDF/DOCX"

**Solution**:
- Ensure the file is not corrupted
- Try converting to TXT format
- Check file permissions

## Configuration Options ⚙️

You can modify these parameters in the code:

```python
# Text chunking
chunk_size = 1000  # Size of each text chunk
chunk_overlap = 200  # Overlap between chunks

# Retrieval
k = 3  # Number of relevant chunks to retrieve

# Ollama
base_url = "http://localhost:11434"  # Ollama server URL
model = "nomic-embed-text"  # Embedding model
```

## API Keys 🔑

### Getting a Google Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and paste it in the sidebar

**Note**: Keep your API key secure and never commit it to version control!

## Performance Tips 💡

- **Chunk Size**: Smaller chunks (500-1000) work better for specific questions
- **Number of Chunks**: Retrieve 3-5 chunks for balanced context
- **Document Quality**: Clean, well-formatted documents yield better results
- **Question Clarity**: Specific questions get more accurate answers

## Security Notes 🔒

- Never share your API keys publicly
- Use environment variables for production deployments
- The chatbot processes documents locally for privacy

## Future Enhancements 🚀

Potential improvements:
- [ ] Support for more file types (CSV, JSON, etc.)
- [ ] Multi-language support
- [ ] Conversation memory across sessions
- [ ] Export chat history
- [ ] Advanced filtering and search
- [ ] User authentication

## Contributing 🤝

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License 📄

This project is open-source and available under the MIT License.

## Support 💬

For issues or questions:
- Check the troubleshooting section
- Review Ollama documentation: [ollama.com/docs](https://ollama.com/docs)
- Review Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)
- Review LangChain documentation: [python.langchain.com](https://python.langchain.com)

## Acknowledgments 🙏

Built with:
- [Streamlit](https://streamlit.io/)
- [Google Gemini](https://ai.google.dev/)
- [Ollama](https://ollama.com/)
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)

---

Made with ❤️ by Your Name
# BEE-RAG
# BEE-RAG
