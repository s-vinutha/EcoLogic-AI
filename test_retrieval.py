import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector

load_dotenv()

# 1. Initialize the SAME model used in ingestion
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# 2. Connect to the SAME collection
# Change 'embedding_function' to 'embeddings'
vector_store = PGVector(
    embeddings=embeddings,  # Updated keyword
    connection=os.getenv("DATABASE_URL"),
    collection_name="ecologic_taxonomy_v2026"
)

# 3. Test a Query
query = "I am looking for eco-friendly shipping boxes"
print(f"🔍 Searching for: {query}")

results = vector_store.similarity_search(query, k=1)

if results:
    print(f"✅ Found Match: {results[0].page_content}")
else:
    print("❌ No match found in the database.")