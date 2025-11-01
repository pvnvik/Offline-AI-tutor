import streamlit as st
import os
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import setup_vectorstore_from_text, get_retriever
from tempfile import NamedTemporaryFile

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        max-width: 80%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .user-message {
        background-color: #2c5282;
        color: #ffffff;
        margin-left: auto;
        margin-right: 2rem;
        border: 1px solid #4299e1;
    }
    .assistant-message {
        background-color: #2d3748;
        color: #ffffff;
        margin-left: 2rem;
        margin-right: auto;
        border: 1px solid #4a5568;
    }
    .file-uploader {
        padding: 2rem;
        border-radius: 0.8rem;
        background-color: #2d3748;
        color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border: 1px solid #4a5568;
    }
    .chat-container {
        padding: 2rem;
        border-radius: 0.8rem;
        background-color: #2d3748;
        color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-top: 2rem;
        border: 1px solid #4a5568;
    }
    /* Additional styles for better visibility */
    .stButton button {
        background-color: #4299e1;
        color: #ffffff;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .stTextInput input {
        background-color: #2d3748;
        color: #ffffff;
        border: 1px solid #4a5568;
    }
    .stMarkdown {
        color: #ffffff;
    }
    .st-emotion-cache-10trblm {
        color: #ffffff;
    }
    .st-emotion-cache-183lzff {
        color: #ffffff;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2d3748;
    }
    section[data-testid="stSidebar"] {
        background-color: #1e1e1e;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

with st.sidebar:
    st.title("📚 Document Upload")
    st.markdown("Upload your study material here")
    
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf'])
    
    if uploaded_file is not None and not st.session_state.file_uploaded:
        try:
            content = uploaded_file.getvalue().decode('utf-8')
            
            file_id = f"upload_{hash(content)}"
            
            vectorstore = setup_vectorstore_from_text(content, file_id)
            st.session_state.retriever = get_retriever(vectorstore)
            st.session_state.file_uploaded = True
            
            st.success("File uploaded and processed successfully!")
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.session_state.file_uploaded = False

st.title("🤖 AI Study Assistant")

for message in st.session_state.messages:
    with st.container():
        st.markdown(
            f"""<div class="chat-message {'user-message' if message['role'] == 'user' else 'assistant-message'}">
                {message['content']}
            </div>""",
            unsafe_allow_html=True
        )

if st.session_state.retriever is None:
    st.warning("Please upload a document first to start chatting!")
else:
    model = OllamaLLM(model="tinyllama")
    template = """You are an expert tutor. The following is study material and a student's question about it. 
    Please provide a clear, detailed, and educational response.

    Study Material:
    {study_material}

    Student Question: {question}

    Please:
    1. Identify the relevant section(s) from the study material
    2. Provide a comprehensive but clear explanation
    3. Include examples where appropriate
    4. Break down complex concepts into simpler terms
    5. If the question relates to technical specifications or protocols, explain their practical applications
    6. Reference specific parts of the study material to support your answer
    7. If the topic involves multiple concepts, show how they interconnect
    8. Suggest follow-up questions the student might want to explore

    Response:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    question = st.chat_input("Ask your question here...")
    
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        
        docs = st.session_state.retriever.invoke(question)
        study_material = "\n\n".join([doc.page_content for doc in docs])
        
        with st.spinner("Thinking..."):
            result = chain.invoke({"study_material": study_material, "question": question})
        
        st.session_state.messages.append({"role": "assistant", "content": result})
        
        st.rerun()