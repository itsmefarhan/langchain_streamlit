import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain # retrieve doc and pass to retrieval chain
from langchain.chains.retrieval import create_retrieval_chain # pass doc to llm
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
system_prompt = (
    'You are an assistant for question-answering tasks. '
    'Use the following pieces of retrieved context to answer '
    'the question. If you don\'t know the answer, say that you '
    'don\'t know. Use three sentences maximum and keep the '
    'answer concise.'
    '\n\n'
    '{context}'
)
prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('human', '{input}')
])

# create chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# pass input to retriever and return relevant part of doc to chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

st.header('Chat with a document')

question = st.chat_input('Your question: ')
if question:
    with st.chat_message('user'):
        st.markdown(question)

    response = rag_chain.invoke({'input': question})
    with st.chat_message('assistant'):
        st.write(response['answer'])