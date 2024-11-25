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
qa_pipeline = ""

def setup1():

    # Load the question-answering pipeline
    qa_pipeline = pipeline("question-answering", model='bert-base-uncased')
    
    # Define the context related to GeeksforGeeks
    context = f"{mycontext}"






json_url = 'https://api.npoint.io/03cc552f40aca75a2bf1'
response = requests.get(json_url)
json_data = response.content

urls = []


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

def main():
    load_dotenv()
    HUGGINGFACEHUB_API_TOKEN= os.environ["HUGGINGFACEHUB_API_TOKEN"]

    st.title("HuggingFace Project")
    mycontext = upload_ivieAi()
    def setup1()

if __name__ == '__main__':
    main()
