import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

apiKey = os.getenv('GOOGLE_GEMINI_KEY')

st.header('Blog Generator')
st.write('Provide the topic of your interest to get the desired article')

topic = st.text_input('Enter your topic')

title_temp = PromptTemplate(input_variables=['topic'], template='Give me an article title on {topic}')
article_temp = PromptTemplate(input_variables=['title'], template='Give me an article on {title}. It should be minimum 3 and 5 paragraphs')

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=apiKey)

title_chain = title_temp | llm | StrOutputParser()
article_chain = article_temp | llm

overall_chain = title_chain | article_chain

if topic:
    response = overall_chain.invoke(topic)
    st.write(response.content)