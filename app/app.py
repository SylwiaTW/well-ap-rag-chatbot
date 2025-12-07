import streamlit as st
import os
from dotenv import load_dotenv

from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

# =============================
#   Stałe konfiguracyjne
# =============================
CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_MODEL")
TOP_K = 3

# Azure credentials
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
EMBED_MODEL = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

# Azure clients
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-02-01"
)

search_client = SearchClient(
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_INDEX,
    AzureKeyCredential(AZURE_SEARCH_KEY)
)


# =============================
#   RAG HELPER FUNCTIONS
# =============================
def embed_query(text):
    res = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return res.data[0].embedding


def retrieve_chunks(query_embedding, k=TOP_K):
    results = search_client.search(
        search_text=None,
        vector_queries=[
            {
                "kind": "vector",
                "vector": query_embedding,
                "fields": "content_vector",
                "k": k
            }
        ]
    )

    chunks = []
    for r in results:
        chunks.append(r["content"])

    return "\n\n---\n\n".join(chunks)


def answer_question(query):
    embedding = embed_query(query)
    context = retrieve_chunks(embedding)

    prompt = f"""
You are a WELL Building Standard assistant.
Answer ONLY using the context below.
If the answer is missing, respond "Not found in WELL v2."

Question:
{query}

Context:
{context}

Answer:
"""

    res = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content


# =============================
#   STREAMLIT APP
# =============================
st.title("🏢 WELL Standard Chatbot")

question = st.text_area("Ask a question about WELL v2:")

if st.button("Ask"):
    with st.spinner("Thinking..."):
        answer = answer_question(question)

        st.markdown("### 🟦 Answer")
        st.write(answer)
