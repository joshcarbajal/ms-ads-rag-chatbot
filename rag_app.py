import streamlit as st
import json
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

# Load API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Load vector store
@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_dir = "faiss_index"
    index_file = os.path.join(index_dir, "index.faiss")

    if os.path.exists(index_file):
        return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        with open("scraped_ms_ads_data_v3.json", "r", encoding="utf-8") as f:
            json_data = json.load(f)

        documents = []
        for item in json_data:
            content = item.get("content", "")
            metadata = {k: v for k, v in item.items() if k != "content"}
            documents.append(Document(page_content=content, metadata=metadata))

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(index_dir)

        return vectorstore

# Load QA chain
@st.cache_resource
def load_chain():
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOpenAI(
        temperature=0.2,
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_api_key
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False  # ⛔ Hide context documents
    )

# Initialize session history
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# UI Layout
st.set_page_config(page_title="UChicago MS-ADS RAG Chatbot", layout="wide")
st.title("🎓 MS in Applied Data Science Q&A Chatbot")
st.markdown("Ask any question about the program, curriculum, admissions, or outcomes.")

# User input with state control
st.text_input("Enter your question:", key="user_query")

if st.session_state.user_query:
    with st.spinner("Generating answer..."):
        qa_chain = load_chain()
        result = qa_chain.run(st.session_state.user_query)

        # Save to chat history
        st.session_state.qa_history.append((st.session_state.user_query, result))

        # Clear input box after response
        st.session_state.user_query = ""

# Display previous Q&A
qa_history = st.session_state.qa_history
labeled_history = [(f"Q{i+1}", f"A{i+1}", q, a) for i, (q, a) in enumerate(qa_history)]

if labeled_history:
    st.subheader("💬 Chat History")
    for q_label, a_label, q, a in reversed(labeled_history):
        st.markdown(f"**{q_label}:** {q}")
        st.markdown(f"**{a_label}:** {a}")

