import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain # retrieve doc and pass to retrieval chain
from langchain.chains.retrieval import create_retrieval_chain # pass doc to llm
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import chromadb.api
chromadb.api.client.SharedSystemClient.clear_system_cache()

apiKey = os.getenv('GOOGLE_GEMINI_KEY')

# define llm
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=apiKey)

# load text
loader = TextLoader('./story.txt')
documents = loader.load()

# create text splitter and split docs into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# convert document chunks into vectors, and retrieve
embeddings = GoogleGenerativeAIEmbeddings(google_api_key=apiKey, model='models/embedding-001')
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()

# create prompt
contextualize_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# create chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# pass input to retriever and return relevant part of doc to chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

history = StreamlitChatMessageHistory()

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer',
)

st.header('Chat with a document')

for message in st.session_state['langchain_messages']:
    role = 'user' if message.type == 'human' else 'assistant'
    with st.chat_message(role):
        st.markdown(message.content)

question = st.chat_input('Your question: ')
if question:
    with st.chat_message('user'):
        st.markdown(question)
    answer_chain = conversational_rag_chain.pick("answer")
    response = answer_chain.stream(
        {'input': question}, config={'configurable': {'session_id': 'any'}}
    )
    with st.chat_message('assistant'):
        st.write_stream(response)