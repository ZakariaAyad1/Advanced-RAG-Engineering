from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 1. Mock Dataset: Intentionally redundant to test MMR efficacy
texts = [
    "LangChain is a framework for developing applications powered by language models.", # D1
    "LangChain enables the creation of complex LLM workflows and chains.",           # D2 (Redundant to D1)
    "Vector databases like Pinecone and Chroma are essential for efficient RAG.",    # D3 (Diverse)
    "Semantic search relies on high-dimensional vector embeddings."                  # D4 (Diverse)
]

# 2. Initialize Embeddings and Vector Store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts, embeddings)

# 3. Configure MMR Retrieval
# 'fetch_k' is the initial pool of candidates to evaluate for diversity.
# 'k' is the final number of documents returned to the LLM context.
# 'lambda_mult' represents the λ parameter (0.0 to 1.0).
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2, 
        "fetch_k": 4, 
        "lambda_mult": 0.5 # Balanced relevance and diversity
    }
)

# 4. Execute Query
query = "What is LangChain and how does it relate to vector DBs?"
docs = retriever.invoke(query)

# 5. Output Results
print(f"Query: {query}\n")
for i, doc in enumerate(docs):
    print(f"Rank {i+1}: {doc.page_content}")

"""
EXPECTED BEHAVIOR:
Standard similarity search might return D1 and D2 because they both contain "LangChain".
MMR with λ=0.5 will likely return D1 (Highest relevance) and D3 or D4 (Highest novelty),
effectively filtering out the redundant information in D2.
"""
