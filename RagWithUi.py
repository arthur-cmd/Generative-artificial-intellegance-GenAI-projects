import ollama
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import streamlit as st
from langchain.schema import AIMessage
from chromadb.config import Settings  # Import settings for Chroma configuration


# Streamlit app setup
st.set_page_config(page_title="Credable.ai", page_icon="🤖")
st.title("Credable.ai")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Welcome to Credable. How can I help you?"),
    ]

# Load website data
website = 'https://docs.google.com/document/d/e/2PACX-1vRLuOf0XBJ1CUXsQrPVnqOp-yLYWpdDQoRYRf3YqOKKRmLs9uzRS6idpLXsJKaiGEFliGNWj0918ifW/pub'
loader = WebBaseLoader(web_paths=(f"{website}",))
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Configure Chroma client settings
client_settings = Settings(
 persist_directory="./chroma_data",  # Directory to store database files
 anonymized_telemetry=False         # Optional: Turn off telemetry if desired
)


# Create Ollama embeddings and vector store
embeddings = OllamaEmbeddings(model="llama3.1")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, client_settings=client_settings)

# Define helper functions
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

retriever = vectorstore.as_retriever()

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

# Streamlit input for user questions
user_question = st.text_input("Enter your question:", "")
if st.button("Submit"):
    if user_question.strip():
        # Process the question
        response = rag_chain(user_question)
        st.session_state.chat_history.append(AIMessage(content=response))
        st.write(f"**Question:** {user_question}")
        st.write(f"**Response:** {response}")
    else:
        st.warning("Please enter a valid question.")

# Display chat history
st.write("### Chat History")
for message in st.session_state.chat_history:
    st.write(f"- {message.content}")
