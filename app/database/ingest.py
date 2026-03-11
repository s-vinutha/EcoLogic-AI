import json
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector

load_dotenv()

# 1. Load Taxonomy
with open("data/taxonomy.json", "r") as f:
    data = json.load(f)

# 2. Format for Embedding
docs = [f"Category: {c['name']}. Sub: {', '.join(c['sub_categories'])}. Filters: {', '.join(c['sustainability_filters'])}" 
        for c in data["categories"]]

# 3. Ingest into pgvector using the most stable model name for 2026
# Remove "models/" prefix manually if the library adds it
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# 4. Use a FRESH collection name 
# (This avoids errors if previous failed runs created mismatched table structures)
collection_name = "ecologic_taxonomy_v1"

vector_store = PGVector.from_texts(
    texts=docs,
    embedding=embeddings,
    connection=os.getenv("DATABASE_URL"),
    collection_name="ecologic_taxonomy_v2026",
    use_jsonb=True  # Better for structured sustainable data
)

print("✅ Taxonomy Ingested successfully into EcoLogic-AI!")