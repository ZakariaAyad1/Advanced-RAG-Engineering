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














# Technical Brief: Maximal Marginal Relevance (MMR) in RAG Pipelines

# 1. Concept: Beyond Simple Similarity

# Standard vector retrieval often suffers from informational redundancy. If a query matches a cluster of near-identical documents, the top-$k$ results will offer repetitive context, wasting the LLM's context window.

# Maximal Marginal Relevance (MMR) is a reranking strategy designed to optimize for both relevance (alignment with the query) and diversity (novelty relative to already selected documents).

# 2. The MMR Mathematical Framework

# The algorithm iteratively selects a document $d$ from the set of candidates $C$ that maximizes the following expression:

# $$\text{MMR} = \arg \max_{d \in C \setminus S} \left[ \lambda \cdot \text{Sim}_1(d, q) - (1 - \lambda) \cdot \max_{s \in S} \text{Sim}_2(d, s) \right]$$

# Key Parameters:

# $q$: The user's query vector.

# $d$: A candidate document not yet selected.

# $S$: The set of documents already selected for the final result.

# $\lambda$ (Lambda): The "Diversity-Relevance" dial (range: 0 to 1).

# $\lambda = 1$: Standard semantic search (maximum relevance).

# $\lambda = 0$: Maximum diversity (ignores the query, picks the most different items).

# $\lambda = 0.5$: An equitable balance between the two.

# 3. Step-by-Step Computational Logic

# Step 1: Seed Selection

# The process begins by selecting the document with the absolute highest cosine similarity to the query. This document becomes the first member of the selected set ($S$).

# Step 2: Iterative Diversity Scoring

# For the remaining candidates, the algorithm calculates:

# Relevance Score: Similarity between the candidate and the query.

# Redundancy Penalty: The maximum similarity between the candidate and any document already in $S$.

# Step 3: Selection

# The document that achieves the highest "Marginal Relevance" (Relevance minus Penalty) is moved to $S$. This continues until the desired number of documents ($k$) is reached.

# 4. Practical Implications for AI Engineers

# Context Window Optimization: MMR ensures that each token in the prompt provides unique, non-overlapping information.

# Hallucination Mitigation: By providing diverse viewpoints or facets of a topic, the LLM is less likely to be trapped in a biased or narrow interpretation caused by redundant retrieved snippets.

# Hyperparameter Tuning: Finding the "Goldilocks" $\lambda$ value is project-specific. For technical manuals, $\lambda \approx 0.7$ is common; for creative or exploratory tasks, a lower $\lambda$ may be preferable.

# 5. Conclusion

# MMR transforms the retriever from a simple "lookup tool" into an intelligent "curator." It ensures that the documents retrieved for the RAG prompt are not just "more of the same," but a representative and comprehensive subset of the knowledge base.
