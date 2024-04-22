import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS  # type: ignore
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from constants import openai_key # type: ignore
from docx import Document
import os

os.environ["OPENAI_API_KEY"]=openai_key

def get_pdf_text(pdf_docs):
    print("Inside get_pdf_text")
    try:
            text=[]
        # for pdf in pdf_docs:
            pdf_reader= PdfReader(pdf_docs)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            
            print('File {pdf} processed'.format(pdf=pdf_docs.name))
    except Exception as e:
        print("Could not read the file:", e)
    return  '\n'.join(text)

def get_wordoc_text(word_docs):
    print("Inside get_wordoc_text")
    try:
            text=[]
        # for doc in word_docs:
            doc_reader= Document(word_docs)
            for para in doc_reader.paragraphs:
                text.append(para.text)

            print('File {doc} processed'.format(doc=word_docs.name))
        # print(text)
    except Exception as e:
        print("Could not read the file:", e)
    return  '\n'.join(text)

# def get_pdf_text_pypdf(pdf_docs):
#     print("Inside get_pdf_text_pypdf")
#     # text=""
#     for pdf in pdf_docs:
#         pdf_reader= PyPDFLoader(pdf)
#         pages = pdf_reader.load_and_split()
#         # for page in pages:
#         #     text+= page.extract_text()
#     return  pages

def get_text_chunks(text):
    print("Inside get_text_chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    print("inside get_vector_store")
    try:
        embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings_model)
        # vector_store=FAISS.from_documents(text_chunks, embedding=embeddings_model)
        vector_store.save_local("faiss_index")
    except Exception as e:
        print("Could not store the file:", e)

# def merge_vector_store(text_chunks, vector_store):
#     print("inside merge_vector_store")
#     try:
#         embeddings_model = OpenAIEmbeddings()
        # vector_store = FAISS.from_texts(text_chunks, embedding=embeddings_model)
        # vector_store=FAISS.from_documents(text_chunks, embedding=embeddings_model)
        # vector_store.save_local("faiss_index")


    # except Exception as e:
    #     print("Could not store the file:", e)

# def get_vector_store_pages(pages):
#     print("Inside get_vector_store")
#     # embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001") # type: ignore
#     embeddings_model = OpenAIEmbeddings()
#     vector_store = FAISS.from_documents(pages, embedding=embeddings_model)
#     # vector_store=FAISS.from_documents(text_chunks, embedding=embeddings_model)
#     vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # model = ChatGoogleGenerativeAI(model="gemini-pro",
    #                          temperature=0.3)

    model = OpenAI(temperature=0.8,tiktoken_model_name='gpt-4')

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')
    
    new_db = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using OpenAI")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        file_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                
                if (file_docs) : 
                    raw_text=''                   
                    for doc in file_docs:
                        doc_name=(doc.name)
                        print('inside main '+ doc_name)
                        doc_type=(doc_name.split('.')[-1])

                        if doc_type=='docx':
                            raw_text += get_wordoc_text(doc)
                        
                            
                        elif doc_type=='pdf':
                            raw_text += get_pdf_text(doc)
                          
                        
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)

                st.success("Done")


if __name__ == "__main__":
    main()