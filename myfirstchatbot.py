import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = "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"
#Upload PDF files
st.header("My first Chatbot")

with  st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(" Upload a PDf file and start asking questions", type="pdf")

#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        # st.write(text)

#Break it into chunks
text_splitter= RecursiveCharacterTextSplitter(
   separators="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
    )
chunks= text_splitter.split_text(text)
# st.write(chunks)

#Generating embeddings
embeddings= OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

#Creating vector store- FAISS
vector_store= FAISS.from_texts(chunks,embeddings)

#Get user question
user_question=st.text_input("Type your question here")

#Do similarity search
if user_question:
    match= vector_store.similarity_search(user_question)
st.write(match)

#define the LLM
lim= ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, max_tokens=1000,model_name="gpt-3.5-turbo")

#output results
chain=load_qa_chain(lim, chain_type="stuff")

#that sorta wraps up your first ever generative AI chatbot in python