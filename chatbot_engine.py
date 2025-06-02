import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import streamlit as st

# Load OpenAI API key from Streamlit Secrets

from openai import OpenAI
client = OpenAI(api_key=st.secrets["openai_api_key"])

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and split law texts
def load_texts_from_data_folder():
    texts, sources = [], []
    for file in ["ra_12009.txt", "irr_ra_12009.txt"]:
        with open(file, "r", encoding="utf-8") as f:
            raw = f.read()
            paragraphs = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 50]
            texts.extend(paragraphs)
            sources.extend([file] * len(paragraphs))
    return texts, sources

texts, sources = load_texts_from_data_folder()
embeddings = model.encode(texts, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Semantic search function
def semantic_search(query, top_k=3):
    q_vec = model.encode([query], convert_to_numpy=True)
    _, idx = index.search(q_vec, top_k)
    return [(sources[i], texts[i]) for i in idx[0]]

# Use GPT to generate a chatbot-style answer
def ask_chatbot(query, context_sections):
    context = "\n\n".join(context_sections)
    prompt = f"""
You are ProcureBot PH, a helpful legal chatbot for the Philippines' Government Procurement Law (RA 12009) and its IRR.

Use the excerpts below to answer the user question clearly, citing legal references when appropriate.

QUESTION:
{query}

RELEVANT LAW SECTIONS:
{context}

ANSWER:
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a legal assistant helping users understand RA 12009 and its IRR."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=600
    )
    return response["choices"][0]["message"]["content"]
