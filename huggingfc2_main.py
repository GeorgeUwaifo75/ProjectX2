import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv
load_dotenv()


#os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

st.title("HuggingFace Project")
st.write("HF:",HUGGINGFACEHUB_API_TOKEN)
