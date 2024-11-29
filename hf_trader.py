import streamlit as st
import os
from dotenv import load_dotenv

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from huggingface_hub import login

load_dotenv()

login("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")

st.write("The Predictor")

#PROMPT = "[INST]YOUR PROMPT HERE[/INST]"
PROMPT = "[INST]When can I buy ethereum?[/INST]"
MAX_LENGTH = 32768  # Do not change
DEVICE = "cpu"


model_id = "agarkovv/CryptoTrader-LM"
base_model_id = "mistralai/Ministral-8B-Instruct-2410"

model = AutoPeftModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

model = model.to(DEVICE)
model.eval()
inputs = tokenizer(
    PROMPT, return_tensors="pt", padding=False, max_length=MAX_LENGTH, truncation=True
)
inputs = {key: value.to(model.device) for key, value in inputs.items()}

res = model.generate(
    **inputs,
    use_cache=True,
    max_new_tokens=MAX_LENGTH,
)
output = tokenizer.decode(res[0], skip_special_tokens=True)
st.write(output)
