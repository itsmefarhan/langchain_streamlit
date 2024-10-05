import streamlit as st

yt_page = st.Page(page='views/yt.py', title='Chat with youtube')
chatbot_page = st.Page(page='views/chatbot.py', title='Ask AI')
chat_page = st.Page(page='views/chat.py', title='Chat with document')
home_page = st.Page(page='views/home.py', title='Blog Generator', default=True)

pg = st.navigation(pages=[home_page, chatbot_page, chat_page, yt_page])

pg.run()
