import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

st.title('Chat with AI')

apiKey = os.getenv('GOOGLE_GEMINI_KEY')

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=apiKey)

prompt = ChatPromptTemplate.from_messages([
    ('system', '''You are an AI chatbot having a conversation with a human. 
     Use the following context to understand the human question. 
     Do not include emojis in your answer. '''),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}')
])

chain = prompt | llm

history = StreamlitChatMessageHistory()

# inject history stored in streamlit into chain
chain_with_history = RunnableWithMessageHistory(
    chain, lambda session_id: history, 
    input_messages_key='input', history_messages_key='chat_history'
)

# question = st.text_input('Your question')
question = st.chat_input('Your question')
if question:
    with st.chat_message('user'):
        st.markdown(question)
    # response = chain_with_history.invoke(
    #     {'input': question}, config={'configurable': {'session_id': 'any'}}        
    # )
    response = chain_with_history.stream(
        {'input': question}, config={'configurable': {'session_id': 'any'}}        
    )
    # st.write(response.content)
    with st.chat_message('assistant'):
        st.write_stream(response)

    # display history
    for message in st.session_state['langchain_messages']:        
        # if message.type == 'human':
        #     st.write('Question: ' + message.content)
        # else:
        #     st.write('Answer: ' + message.content)
        role = 'user' if message.type == 'human' else 'assistant'
        with st.chat_message(role):        
            st.markdown(message.content)