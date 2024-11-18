import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = hf_token

# Initialize LLM and embeddings
llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Data processing
loader = WebBaseLoader("https://www.mapcommunications.com/call-center-faq/")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_docs = text_splitter.split_documents(docs)

db = Chroma.from_documents(final_docs, embeddings)
retriever = db.as_retriever()

# History-aware retriever
q_prompt = """
Given a chat history and the latest user question 
which might reference context in the chat history,
formulate a standalone question which can be understood 
without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is.
"""

prompt2 = ChatPromptTemplate.from_messages([
    ("system", q_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt2)

# QA chain
system_prompt = (
    """ You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer
    the question. If you don't know the answer, say that you 
    don't know. Use three sentences maximum and keep the 
    answer concise.
    """
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, chain)

# Chat history management
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Streamlit UI
st.set_page_config(page_title="Chatbot", layout="wide")
st.title("Call Center FAQ Chatbot")

session_id = st.session_state.get("session_id", "default_session")
chat_history = st.session_state.get("chat_history", [])

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

# Display chat
for msg in st.session_state.chat_messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")

# User input
user_input = st.text_input("Ask a question:", key="input")
if st.button("Send"):
    if user_input:
        # Append user message to chat
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        
        # Get response
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )["answer"]
        
        # Append bot response to chat
        st.session_state.chat_messages.append({"role": "bot", "content": response})
        st.experimental_rerun()
