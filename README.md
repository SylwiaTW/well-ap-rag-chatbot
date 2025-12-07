# WELL AP RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the **WELL Building Standard v2** using Azure OpenAI and Azure AI Search.  
The system retrieves verified text fragments from the WELL Standard and uses them as strict context for generating compliant responses.

---

## Project Goal

The goal of this project is to build an **Intelligent WELL AP Assistant** that helps users understand WELL feature requirements by querying a chatbot.  
The assistant retrieves content from the WELL Building Standard, ensuring every answer is grounded in the official documentation.

---

## System Architecture

### 1. Data Processing Pipeline

Implemented in `process_data.py`.

### 2. Streamlit Chat Application

Implemented in `app.py`.

---

## Project Structure

```
project/
│
├── data/
│   ├── WELL-Building-Standard-wellv2.pdf
│   ├── chunks.jsonl
│   ├── embeddings.jsonl
│
├── process_data.py
├── app.py
├── .env
└── README.md
```

---