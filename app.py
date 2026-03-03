import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit UI
st.title(":green[RAG Chatbot]")
st.subheader(":blue[Tips]")
st.write("""
Follow the steps to use this application:
* Upload your PDF document in the sidebar.
* Write your query and start chatting with the bot.
""")

# Sidebar for PDF upload
file_uploaded = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if file_uploaded:
    # Extract text from PDF
    reader = PdfReader(file_uploaded)
    file_text = "".join([page.extract_text() for page in reader.pages])

    # Step 1: Configure Gemini
    key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=key)

    # ✅ Use correct model name with "models/" prefix
    # Run list_models() once to confirm available models in your environment
    llm_model = genai.GenerativeModel("gemini-3-flash-preview")

    # Step 2: Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-V2')

    # Step 3: Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(file_text)

    # Step 4: Create FAISS vector store
    vector_store = FAISS.from_texts(chunks, embedding_model)

    # Step 5: Configure retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Function to generate response
    def generate_response(query: str) -> str:
        retrieved_docs = retriever.invoke(query)  # ✅ Correct method
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"""
        You are a helpful assistant answering questions based on the following context:
        {context}
        User query: {query}
        """
        content = llm_model.generate_content(prompt)
        return content.text

    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display chat history
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.write(f':green[User:] :blue[{msg["text"]}]')
        else:
            st.write(f':orange[Chatbot:] {msg["text"]}')

    # Chat input form
    with st.form('Chat Form', clear_on_submit=True):
        user_input = st.text_input('Enter Your Text Here:')
        send = st.form_submit_button('Send')

    # Handle user input
    if user_input and send:
        st.session_state.history.append({"role": 'user', "text": user_input})
        model_output = generate_response(user_input)
        st.session_state.history.append({'role': 'chatbot', 'text': model_output})
        st.rerun()