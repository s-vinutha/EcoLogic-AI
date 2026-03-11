from langchain_google_genai import ChatGoogleGenerativeAI
from app.schemas.product import ProductAnalysis

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def generate_tags(product_name: str, context: str):
    prompt = f"""
    You are an expert in Sustainable Commerce. 
    Use the following verified taxonomy context: {context}
    
    Categorize this product: {product_name}
    Return a structured JSON matching the ProductAnalysis schema.
    """
    return llm.with_structured_output(ProductAnalysis).invoke(prompt)