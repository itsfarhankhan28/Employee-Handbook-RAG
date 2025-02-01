# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
import streamlit as st

#Two main components to create a RAG application
# 1) Indexing
# 2) Retrieval

# First step of indexing
# selecting chat model, embedding model and vector store.

# Chat model

llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0.3,
)

#Embedding model

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

#Vector store

vector_store = Chroma(
    collection_name="rag_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

#Load document
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('./Employee-Handbook.pdf')
docs = loader.load()
# print(docs)

#Splitting text
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
documents = text_splitter.split_documents(docs)

# Index chunks
document_ids = vector_store.add_documents(documents=documents)

prompt = hub.pull("rlm/rag-prompt")

#Define the state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retriever(state: State):
    document_content = vector_store.similarity_search(state["question"])
    return {"context":document_content}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retriever, generate])
graph_builder.add_edge(START, "retriever")
graph = graph_builder.compile()

# response = graph.invoke({"question": "What will be the working hours for the employee"})
# print(response["answer"])

st.title("RAG Chatbot with Streamlit")
st.write("Enter your query below to interact with the system.")

query = st.text_input("Ask a question:")
if st.button("Get Answer") and query:
    response = graph.invoke({"question": query})
    st.subheader("Response:")
    st.write(response["answer"])