import os
import streamlit as st
import transformers
from transformers import pipeline
import pandas as pd
import requests
import json
import urllib.request

from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv

mycontext=""
context = ""
qa_pipeline = ""

json_url = 'https://api.npoint.io/03cc552f40aca75a2bf1'
response = requests.get(json_url)
json_data = response.content

urls = []


def setup1(mycontext):
    # Load the question-answering pipeline
    qa_pipeline = pipeline("question-answering", model='bert-base-uncased')
    
    # Define the context related to GeeksforGeeks
    context = f"{mycontext}"


#Upload IvieAI dataset
def upload_ivieAi():
    # Load the JSON data into a Python dictionary
    data = json.loads(json_data)

    # Extract the first "reply" values from each item in "allpushdata"
    text = ""
    for item in data["allpushdata"]:
        first_reply = item["replies"][0]["reply"]
        text += first_reply + "\n"
    
    return text

def handle_userinput(text):
    result = qa_pipeline(question=text, context=context)
    st.write(f"Q: {text}\nA: {result['answer']}\n")


def main():
    load_dotenv()
    HUGGINGFACEHUB_API_TOKEN= os.environ["HUGGINGFACEHUB_API_TOKEN"]

    st.title("HuggingFace Project")
    mycontext = upload_ivieAi()
    setup1(mycontext)

    user_question = st.text_input("Ask a question about your documents:")
    # Ask a question
    if user_question:
        handle_userinput(user_question)
    

if __name__ == '__main__':
    main()
