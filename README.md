# Offline AI Tutor

An intelligent tutoring system that leverages local Language Models and vector embeddings to provide interactive learning assistance with uploaded study materials—completely offline and privacy-focused.

## 🤔 Why Offline-AI-tutor?

This project facilitates personalized, offline-accessible AI tutoring by integrating Retrieval-Augmented Generation (RAG) with local language models. Key highlights:

- **Vector Search**: Efficiently processes and queries document embeddings with Chroma DB for accurate similarity-based retrieval.
- **Context-Aware Chunking**: Uses recursive character splitting to keep chunks within embedding model context limits.
- **Offline Functionality**: Seamless learning experience without needing an internet connection.
- **Context-Aware Responses**: Combines retrieved study material with an AI tutor prompt to generate pedagogical, structured explanations.
- **Interactive Web & CLI Interfaces**: Choose between a modern Streamlit web UI and a command-line interface.
- **Privacy First**: All documents, embeddings, and model inferences stay 100% local on your machine.

---

## 🌟 Features

- 📚 **Multi-Format Document Support**: Upload and process `.pdf` and `.txt` files with automatic multi-encoding detection (`utf-8`, `utf-8-sig`, `latin-1`, `cp1252`).
- ✂️ **Smart Text Chunking**: Recursively splits documents with configurable chunk sizes and overlap to fit model context windows.
- 🔍 **Semantic Vector Search**: Powered by Chroma DB and Ollama embeddings (`mxbai-embed-large`).
- 💬 **Interactive Chat Interface**: Context-aware tutoring with clear explanations, examples, and concept breakdowns.
- 🎨 **Modern Dark Theme UI**: Sleek, responsive interface built with Streamlit.
- 🖥️ **CLI Mode**: Terminal-based chat interface via `main.py`.
- 🔒 **100% Offline & Private**: Zero cloud dependencies or data leaks.

---

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Language Model**: Ollama (`tinyllama`)
- **Embeddings**: Ollama (`mxbai-embed-large`)
- **Vector Store**: Chroma DB (`langchain-chroma`)
- **Orchestration**: LangChain (`langchain`, `langchain-ollama`, `langchain-text-splitters`)
- **PDF Extraction**: `pypdf`

---

## 📋 Prerequisites

Before running the application, make sure you have:

1. **Python 3.8+** installed.
2. **[Ollama](https://ollama.com/)** installed and running on your system.
3. Download the required models in Ollama:
   ```bash
   ollama pull tinyllama
   ollama pull mxbai-embed-large
   ```

---

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pvnvik/Offline-AI-tutor.git
   cd Offline-AI-tutor
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Windows (CMD):
   .\venv\Scripts\activate.bat
   # On macOS / Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎯 Usage

### Web Interface (Recommended)

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open your browser at `http://localhost:8501`.
3. Upload your study material (`.pdf` or `.txt`) using the sidebar.
4. Ask questions in the chat input to receive contextual answers from the tutor!

### Command-Line Interface (CLI)

Run the terminal-based tutor:
```bash
python main.py
```

---

## 📁 Project Structure

```
Offline-AI-tutor/
├── app.py              # Streamlit web UI & file upload handler
├── main.py             # CLI interface for interactive questioning
├── vector.py           # Document chunking, embeddings & vector store logic
├── requirements.txt    # Project Python dependencies
└── README.md           # Project documentation
```

---

## 🔧 Configuration

The application components can be easily customized:

1. **Embedding Model & Chunking** ([vector.py](file:///c:/Users/vikrant/Desktop/Offline-AI-Agent/vector.py)):
   - Default embedding model: `mxbai-embed-large`
   - Default chunk size: `800` characters with `120` character overlap (tuned for 512-token context limits)
   - Vector persistence directory: `./vectorstore_[hash]`

2. **LLM & Tutoring Persona** ([app.py](file:///c:/Users/vikrant/Desktop/Offline-AI-Agent/app.py) & [main.py](file:///c:/Users/vikrant/Desktop/Offline-AI-Agent/main.py)):
   - Default LLM: `tinyllama` (can be switched to `llama3`, `mistral`, `phi3`, etc.)
   - Customizable tutoring prompt template with pedagogical guidelines.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://ollama.com/)
- [Streamlit](https://streamlit.io/)
- [Chroma DB](https://www.trychroma.com/)