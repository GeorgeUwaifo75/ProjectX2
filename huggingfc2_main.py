import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv
load_dotenv()


HUGGINGFACEHUB_API_TOKEN= os.environ["HUGGINGFACEHUB_API_TOKEN"]

st.title("HuggingFace Project")
#st.write("HF:",HUGGINGFACEHUB_API_TOKEN)

llm =  HuggingFaceEndpoint(repo_id="mistralai/Mistral-Nemo-Instruct-2407")

out = llm.invoke("Write a short poem on India")
#st.write(out)
