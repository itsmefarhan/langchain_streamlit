import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SimpleSequentialChain

apiKey = os.getenv('GOOGLE_GEMINI_KEY')
print(apiKey)
st.header('Blog Generator')
st.write('Enter the topic of your interest to get the desired article')


topic = st.text_input('Enter your topic')
title_temp = PromptTemplate(input_variables=['topic'], template='Give me an article title on {topic}')
article_temp = PromptTemplate(input_variables=['content'], template='Give me an article on {content}. It should be minimum 3 and 5 paragraphs')
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=apiKey)

title_chain = LLMChain(llm=llm, prompt=title_temp, verbose=True)
article_chain = LLMChain(llm=llm, prompt=article_temp, verbose=True)
overall_chain = SimpleSequentialChain(chains=[title_chain, article_chain], verbose=True)

if topic:
    response = overall_chain.run(topic)
    st.write(response)