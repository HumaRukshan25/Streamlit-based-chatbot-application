import streamlit as st
import pickle
import openai
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# User authentication
user = st.text_input("Username")
password = st.text_input("Password", type="password")

if user == "Huma Rukshan" and password == "Huma@task1":
    st.sidebar.success("Logged in as {}".format(user))
else:
    st.sidebar.error("Authentication failed")

with st.sidebar:
    st.title('PDF chat app')
    st.markdown("""
     This app is built using <br>
     -[Streamlit](https://streamlit.io/)<br>
     -[LangChain](https://www.langchain.com/)<br>
     -[openAI](https://openai.com/blog/openai-api)
    """, unsafe_allow_html=True)
    st.write("Made by Huma Rukshan")

def main():
    st.header("PDF Chatbot")

    load_dotenv()

    # Upload PDF
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorstore, f)

        query = st.text_input("Ask a question about your PDF")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_document=docs, question=query)
            st.write(response)

if __name__ == '_main_':
    main()
