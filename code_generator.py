import streamlit as st
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from constants import openai_key # type: ignore
from langchain.chains import LLMChain, SimpleSequentialChain
import os

os.environ["OPENAI_API_KEY"]=openai_key

def generate_prompt(user_question, user_language):

    prompt_template = """Act a developer and given any scenario and coding language generate a code. If the scenario is not possible to create in the asked language, then provide alternate language option \n\n
    scenario: \n {scenario}?\n
    language: \n {language}?\n
    answer: 
    """

    model = OpenAI(temperature=0.8,tiktoken_model_name='gpt-4')

    prompt = PromptTemplate(template = prompt_template, input_variables = ["scenario", "language"])

    chain = LLMChain(llm=model, prompt=prompt)

    response = chain.invoke(
        {"scenario":user_question, "language": user_language}
        , return_only_outputs=True)

    return response

def main():
    st.set_page_config("Code Generator")
    st.header("Type your query to write the code")

    user_question = st.text_input("Give any scenario")

    user_language = st.selectbox('Select language',['python','R','Java'])

    st.button("Submit")

    if user_language :
        if user_question:
            response=generate_prompt (user_question, user_language)
            st.text_area(label="code result",value=response['text'],height=500)

if __name__ == "__main__":
    main()