import fitz
import json
import re
import os
from dotenv import load_dotenv

from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

# =============================
#   Stałe konfiguracyjne
# =============================
PDF_PATH = "data/WELL-Building-Standard-wellv2.pdf"
CHUNKS_PATH = "data/chunks.jsonl"
EMBED_PATH = "data/embeddings.jsonl"

FEATURE_REGEX = re.compile(r"\b([A-Z]{1,2}\d{2})\b")

# Azure credentials
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

EMBED_MODEL = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

# Azure OpenAI client
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-02-01"
)


# =============================
# Chunkowanie PDF
# =============================
def chunk_pdf():
    print("📄 Chunking PDF...")
    doc = fitz.open(PDF_PATH)

    chunks = []
    current_feature, current_text, start_page = None, [], None

    for i in range(len(doc)):
        page_no = i + 1
        text = doc[i].get_text()

        match = FEATURE_REGEX.search(text)

        if match:
            if current_feature:
                chunks.append({
                    "feature": current_feature,
                    "page_start": start_page,
                    "page_end": page_no - 1,
                    "content": "\n".join(current_text)
                })

            current_feature = match.group(1)
            current_text = [text]
            start_page = page_no

        else:
            if current_feature:
                current_text.append(text)

    # Final chunk
    if current_feature:
        chunks.append({
            "feature": current_feature,
            "page_start": start_page,
            "page_end": len(doc),
            "content": "\n".join(current_text)
        })

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print("✔ Chunków:", len(chunks))


# =============================
# Embedding chunków
# =============================
def embed_text(text):
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


def embed_chunks():
    print("✨ Generating embeddings...")

    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    with open(EMBED_PATH, "w", encoding="utf-8") as f:
        for i, ch in enumerate(chunks):
            vector = embed_text(ch["content"])

            doc = {
                "id": f"{ch['feature']}_{i}",
                "content": ch["content"],
                "metadata": f"{ch['feature']} pages {ch['page_start']}-{ch['page_end']}",
                "content_vector": vector
            }

            f.write(json.dumps(doc) + "\n")

    print("✔ Embeddingów:", len(chunks))


# =============================
# Upload do Azure Search
# =============================
def upload_to_search():
    print("🚀 Uploading documents to Azure Search...")

    search_client = SearchClient(
        AZURE_SEARCH_ENDPOINT,
        AZURE_SEARCH_INDEX,
        AzureKeyCredential(AZURE_SEARCH_KEY)
    )

    docs = []
    with open(EMBED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    search_client.upload_documents(docs)

    print("✔ Uploaded:", len(docs))


# =============================
# MAIN PIPELINE
# =============================
if __name__ == "__main__":
    chunk_pdf()
    embed_chunks()
    upload_to_search()

    print("\n🎉 Pipeline ready — chatbot can now retrieve WELL chunks!")
