# Offline AI Tutor

An intelligent tutoring system that leverages local Language Models and vector embeddings to provide interactive learning assistance with uploaded study materials.

## 🤔 Why Offline-AI-tutor?

This project aims to facilitate personalized, offline-accessible AI tutors by integrating advanced retrieval techniques with language models. The core features include:

- **Vector Search**: Efficiently process and query document embeddings for quick similarity-based retrieval.

- **Offline Functionality**: Enable seamless learning experiences without internet connectivity.

- **Context-Aware Responses**: Generate detailed, relevant answers by combining study material retrieval with AI models.

- **Interactive Web Interface**: Upload documents, manage content, and engage with the AI tutor effortlessly.

- **Modular Setup**: Easy dependency management with pandas, langchain, streamlit, and more for scalable development.

## 🌟 Features

- 📚 Upload and process study materials (currently supports .txt files)
- 🔍 Semantic search through study content
- 💬 Interactive chat interface with context-aware responses
- 🎨 Modern dark theme UI
- 🚀 Fully offline operation
- 📱 Responsive design
- 🔒 Privacy-focused (all processing happens locally)

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Language Model**: Ollama (tinyllama)
- **Vector Store**: Chroma DB
- **Embeddings**: mxbai-embed-large
- **Framework**: LangChain

## 📋 Prerequisites

Before running the application, make sure you have:

1. Python 3.8 or higher installed
2. Ollama installed and running locally
3. Sufficient disk space for vector storage

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/pvnvik/Offline-AI-tutor.git
cd Offline-AI-tutor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Upload your study material through the sidebar
3. Start asking questions about the content
4. Get contextual, relevant responses from the AI tutor

## 📁 Project Structure

```
Offline-AI-tutor/
├── app.py              # Main Streamlit application
├── vector.py           # Vector store and document processing
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## 🔧 Configuration

The application uses several key components that can be configured:

1. **Vector Store**:
   - Located in `./vectorstore_[hash]`
   - Automatically created per document
   - Persists embeddings for faster subsequent access

2. **Language Model**:
   - Default: tinyllama
   - Can be modified in `app.py`

3. **Embeddings**:
   - Default: mxbai-embed-large
   - Can be modified in `vector.py`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with LangChain
- Powered by Ollama
- UI crafted with Streamlit
- Vector search by Chroma DB

## 💡 Future Improvements

- [ ] Support for PDF and other document formats
- [ ] Multiple document handling
- [ ] Enhanced chat history management
- [ ] Export chat conversations
- [ ] Custom model selection
- [ ] Progress indicators for long operations
- [ ] Improved error handling
- [ ] User preferences storage