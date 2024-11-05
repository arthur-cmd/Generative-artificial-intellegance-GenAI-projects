import ollama
import bs4
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Initialize FastAPI
app = FastAPI()

# Define data model for request
class Query(BaseModel):
    question: str

# Load and process website data
website = 'https://www.marktechpost.com/2024/04/21/coconut-a-high-quality-large-scale-dataset-for-next-gen-segmentation-models/'
loader = WebBaseLoader(
    web_paths=(f"{website}",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("td-post-content tagdiv-type", "td-post-header", "td-post-title")
        )
    ),
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="llama3.1")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Define Ollama LLM function
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Define helper functions for RAG
retriever = vectorstore.as_retriever()

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

# Define FastAPI endpoint for the RAG pipeline
@app.post("/ask")
async def ask_question(query: Query):
    try:
        # Pass the question to the RAG chain
        result = rag_chain(query.question)
        return {"question": query.question, "response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server (if running as a standalone script)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
