from langchain_openai import OpenAIEmbeddings
import numpy as np

# Initialize OpenAI embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Sample documents
documents = [
    "Artificial intelligence is transforming industries.",
    "Machine learning is a subset of artificial intelligence.",
    "The weather is nice today.",
    "Deep learning techniques are improving AI applications.",
    "Stock market trends are influenced by economic policies."
]

# Generate embeddings for documents
document_embeddings = [embeddings.embed_query(doc) for doc in documents]

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# User input for search query
query = input("Enter your search query: ")

# Embed the user query
query_embedding = embeddings.embed_query(query)

# Compute similarities with all documents
similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in document_embeddings]

# Find the most similar document
best_match_index = np.argmax(similarities)
best_match_text = documents[best_match_index]
best_match_score = similarities[best_match_index]

# Display results
print("\nBest matching document:")
print(f'"{best_match_text}" (Score: {best_match_score:.4f})')
