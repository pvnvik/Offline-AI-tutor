from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

def read_text_file(file_path):
    """Read content from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def process_text_content(content, source_id="uploaded_content"):
    """
    Process text content directly into chunks and documents
    Args:
        content: The text content to process
        source_id: Identifier for the source of the content
    Returns:
        tuple: (documents, ids) for vector store creation
    """
 
    chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
    

    documents = []
    ids = []
    
    for i, chunk in enumerate(chunks):
        document = Document(
            page_content=chunk,
            metadata={
                'source': source_id,
                'chunk_id': i,
                'chunk_size': len(chunk),
                'total_chunks': len(chunks)
            }
        )
        documents.append(document)
        ids.append(f"doc_{i}")
        
    return documents, ids

def setup_vectorstore(file_path):
    """
    Set up a new vector store from a file
    Args:
        file_path: Path to the text file to process
    Returns:
        Chroma: Initialized vector store
    """

    content = read_text_file(file_path)
    if not content:
        raise ValueError("Could not read file content")

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
  
    documents, ids = process_text_content(content, file_path)


    db_location = f"./vectorstore_{hash(file_path)}"
    
  
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_location,
        ids=ids
    )
    
    return vectorstore

def setup_vectorstore_from_text(content, source_id="uploaded_content"):
    """
    Set up a new vector store directly from text content
    Args:
        content: The text content to process
        source_id: Identifier for the source of the content
    Returns:
        Chroma: Initialized vector store
    """

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    

    documents, ids = process_text_content(content, source_id)
    

    db_location = f"./vectorstore_{hash(source_id)}"
    

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_location,
        ids=ids
    )
    
    return vectorstore

def get_retriever(vectorstore):
    """
    Create a retriever from a vector store
    Args:
        vectorstore: Chroma vector store instance
    Returns:
        Retriever object
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

if __name__ == "__main__":
    print("Vector store utilities ready for use.")
    file_path = os.environ.get("VECTORSTORE_FILE", "IOT_concepts.txt")
    try:
        vectorstore = setup_vectorstore(file_path)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        print(f"Successfully created/loaded vector store for '{file_path}'")
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        vectorstore = None
        retriever = None
else:
    vectorstore = None
    retriever = None

def query_vectorstore(query: str, k: int = 3):
    """
    Query the vector store for similar documents
    
    Args:
        query (str): The query text
        k (int): Number of results to return (default: 3)
    
    Returns:
        List of tuples containing (Document, relevance_score)
    """
    if vectorstore is None:
        raise ValueError("Vectorstore is not initialized. Call setup_vectorstore(file_path) first.")
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results

def get_relevant_context(query: str, k: int = 3):
    """
    Get relevant context from the vector store based on a query
    
    Args:
        query (str): The query text
        k (int): Number of results to return (default: 3)
    
    Returns:
        str: Concatenated relevant context
    """
    results = query_vectorstore(query, k=k)
    context = []
    
    for doc, score in results:
        context.append(f"[Score: {score:.4f}]\n{doc.page_content}")
    
    return "\n\n".join(context)