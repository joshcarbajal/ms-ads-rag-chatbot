import streamlit as st
import collections
import json
import os

# --- Actual Imports for External Libraries ---
# Ensure these are installed:
# pip install streamlit
# pip install sentence-transformers
# pip install faiss-cpu
# pip install langchain-community
# pip install langchain-core
# pip install langchain-openai
# pip install openai

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document


# --- Configuration ---
FAISS_INDEX_PATH = "faiss_ms_ads_index"
DATA_FILE = "ms_applied_data_science_content.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
LLM_MODEL = "gpt-3.5-turbo" # Or "gpt-4", etc.
TEMPERATURE = 0.0 # For more deterministic answers


# --- Function to load data, create/load FAISS index, and set up RAG chain ---
# Use st.cache_resource to ensure these heavy operations run only once
@st.cache_resource
def setup_rag_system():
    """
    Sets up the RAG system components: loads data, creates/loads FAISS index,
    initializes embeddings, LLM, and the RAG chain.
    """
    print("Setting up RAG system components...")

    # 1. Data Loading
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            scraped_data = json.load(f)
        print(f"Loaded scraped content from {DATA_FILE}")
    except FileNotFoundError:
        st.error(f"Error: {DATA_FILE} not found. Please run the web scraping script first.")
        st.stop() # Stop Streamlit app if data is missing

    # 2. Text Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    all_langchain_documents = []
    for url, content in scraped_data.items():
        if not isinstance(content, str):
            content = str(content)
        
        chunks = text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            all_langchain_documents.append(
                Document(page_content=chunk, metadata={"source_url": url, "chunk_index": i})
            )
    print(f"Total LangChain Documents (chunks) created: {len(all_langchain_documents)}")

    # 3. Embedding Generation
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("HuggingFaceEmbeddings model initialized.")

    # 4. Vector Storage with FAISS
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading FAISS vector store from '{FAISS_INDEX_PATH}'...")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
        print("FAISS vector store loaded.")
    else:
        print("Creating FAISS vector store from documents (this may take a while)...")
        vector_store = FAISS.from_documents(all_langchain_documents, embeddings_model)
        print("FAISS vector store created.")
        print(f"Saving FAISS vector store to '{FAISS_INDEX_PATH}'...")
        vector_store.save_local(FAISS_INDEX_PATH)
        print("FAISS vector store saved.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    print(f"Retriever initialized to fetch {retriever.search_kwargs['k']} documents.")

    # 5. LLM Initialization (OpenAI)
    # Ensure OPENAI_API_KEY is set as an environment variable
    if "OPENAI_API_KEY" not in os.environ:
        st.error("Error: OPENAI_API_KEY environment variable not set.")
        st.stop()
    
    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
    print(f"LLM ({llm.model_name}) initialized.")

    # 6. LangChain RAG Chain Construction
    rag_prompt_template = """
    You are an AI assistant for the University of Chicago's MS in Applied Data Science program.
    Use the following retrieved context to answer the user's question.
    If the answer is not in the context, state that you don't know and do not try to make up an answer.
    Keep the answer concise and to the point.

    Context:
    {context}

    Question: {question}
    Answer:
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
    print("RAG Prompt Template created.")

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    print("LangChain RAG chain constructed.")
    
    return rag_chain, retriever # Return retriever for displaying context


# --- Streamlit UI ---
st.set_page_config(page_title="UChicago MSADS RAG Chatbot", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stExpander {
        border-radius: 10px;
        border: 1px solid #ddd;
        padding: 10px;
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎓 UChicago MS in Applied Data Science Chatbot")
st.markdown("Ask me anything about the **MS in Applied Data Science** program at the University of Chicago!")

# Setup RAG system (cached)
rag_chain, retriever = setup_rag_system()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the RAG chain
                ai_response = rag_chain.invoke(prompt)
                st.markdown(ai_response)
                
                # Optionally display retrieved context for debugging/transparency
                with st.expander("Show Retrieved Context"):
                    st.write("Retrieved documents (top 4 relevant chunks):")
                    retrieved_docs = retriever.invoke(prompt)
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Document {i+1} (Source: {doc.metadata.get('source_url', 'N/A')})**")
                        st.code(doc.page_content[:500] + "...", language="text") # Show first 500 chars

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.warning("Please ensure your `OPENAI_API_KEY` is correctly set and the OpenAI model is accessible.")
                ai_response = "I apologize, but I encountered an error while processing your request. Please try again later."
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

