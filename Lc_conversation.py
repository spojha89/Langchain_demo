import os
from langchain_openai import OpenAI
# from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import SequentialChain
from langchain.chains import LLMChain
import streamlit as st
import os
from constants import openai_key # type: ignore


st.title('first langchain demo')
input=st.text_input('Search about celebrity')

os.environ["OPENAI_API_KEY"]=openai_key


prompt1 = PromptTemplate (
  input_variables= ["name"],  # type: ignore
  template=  "Tell about celebrity {name}"
)

name_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')

dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')

descr_memory = ConversationBufferMemory(input_key='dob', memory_key='chat_history')

llm=OpenAI(temperature=0.8)
chain=LLMChain(llm=llm,prompt=prompt1,verbose=True,output_key='person',memory=name_memory)

second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="when was {person} born"
)

chain2=LLMChain(
    llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob',memory=dob_memory)
# Prompt Templates

third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happened around {dob} in the world"
)
chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='description',memory=descr_memory)
parent_chain=SequentialChain(
    chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'],verbose=True)


if input:
    st.write(parent_chain({'name':input}))

    with st.expander('Person Name'): 
        st.info(name_memory.buffer)

    with st.expander('Major Events'): 
        st.info(descr_memory.buffer)