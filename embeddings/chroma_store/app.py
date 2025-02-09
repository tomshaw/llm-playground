from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import tiktoken

# OpenAI pricing for text-embedding-3-small
COST_PER_1K_TOKENS = 0.00002  # $0.00002 per 1,000 tokens

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# OpenAI tokenizer
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

# Function to count tokens in text
def count_tokens(text):
    return len(tokenizer.encode(text))

# Function to estimate cost based on token count
def estimate_cost(texts):
    total_tokens = sum(count_tokens(text) for text in texts)
    cost = (total_tokens / 1000) * COST_PER_1K_TOKENS
    return total_tokens, cost

# Diverse dataset: Each document has a title and body
documents = [
    {"title": "The Rise of Artificial Intelligence", "body": "Artificial intelligence is transforming industries, automating tasks, and enabling breakthroughs in fields like healthcare and finance."},
    {"title": "The Great Wall of China", "body": "Built over several dynasties, the Great Wall of China stretches over 13,000 miles and served as protection against invasions from the north."},
    {"title": "Quantum Computing Explained", "body": "Unlike classical computers, quantum computers use qubits that can exist in multiple states simultaneously, promising exponential computational power."},
    {"title": "The Stock Market Crash of 1929", "body": "The 1929 crash marked the beginning of the Great Depression, leading to financial ruin for millions and major reforms in financial regulation."},
    {"title": "Benefits of a Healthy Diet", "body": "Eating a balanced diet rich in whole foods, lean proteins, and healthy fats contributes to longevity, mental clarity, and disease prevention."},
    {"title": "The James Webb Space Telescope", "body": "Launched in 2021, the JWST provides deep space imaging capabilities, allowing scientists to study the formation of early galaxies."},
    {"title": "The Psychology of Motivation", "body": "Intrinsic and extrinsic motivation drive human behavior, influencing everything from workplace productivity to personal goals."},
    {"title": "The Evolution of Video Games", "body": "From Pong to VR gaming, video games have evolved into an entertainment industry worth billions, blending storytelling and interactivity."},
    {"title": "The Impact of Climate Change", "body": "Rising global temperatures lead to extreme weather events, rising sea levels, and biodiversity loss, urging the need for sustainable practices."},
    {"title": "The History of the Olympics", "body": "Originating in ancient Greece, the Olympic Games have become the world’s premier sporting event, uniting nations through competition."}
]

# Format data as LangChain Documents
document_objs = [Document(page_content=f"{doc['title']}\n{doc['body']}") for doc in documents]

# Calculate cost of storing documents
doc_texts = [f"{doc['title']} {doc['body']}" for doc in documents]
doc_tokens, doc_cost = estimate_cost(doc_texts)
print(f"\n📄 Estimated cost to store documents: ${doc_cost:.6f} ({doc_tokens} tokens)")

# Store embeddings in Chroma (cost incurred here)
vectorstore = Chroma.from_documents(document_objs, embeddings)

# User input search query
query = input("\n🔍 Enter your search query: ")

# Calculate cost of processing the query
query_tokens, query_cost = estimate_cost([query])
print(f"💰 Estimated cost for query: ${query_cost:.6f} ({query_tokens} tokens)")

# Search for similar documents
results = vectorstore.similarity_search(query, k=1)

# Display result
print("\n📌 Best matching document:")
print(f"📖 Title: {results[0].page_content.splitlines()[0]}")
print(f"📝 Summary: {results[0].page_content.splitlines()[1]}")

# Total cost summary
total_cost = doc_cost + query_cost
print("\n💲 **Cost Summary** 💲")
print(f"📄 Documents Cost: ${doc_cost:.6f} ({doc_tokens} tokens)")
print(f"🔍 Query Cost: ${query_cost:.6f} ({query_tokens} tokens)")
print(f"💰 **Total Estimated Cost:** ${total_cost:.6f}")
