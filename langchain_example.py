import os
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
import streamlit as st
import os
from constants import openai_key # type: ignore

st.title('first langchain demo')
input=st.text_input('Search here')

os.environ["OPENAI_API_KEY"]=openai_key
llm=OpenAI(temperature=0.8)

if input:
  st.write(llm.invoke(input))