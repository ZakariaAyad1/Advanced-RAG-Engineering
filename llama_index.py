"""
Advanced RAG Implementation with LlamaIndex
===========================================
This module demonstrates a production-grade RAG system using LlamaIndex,
covering document ingestion, chunking, embedding, retrieval, and query processing.
"""

import os
from pathlib import Path
from typing import List, Optional

# Core LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,           # Main index for vector storage
    SimpleDirectoryReader,      # Document loader
    StorageContext,             # Storage configuration
    Settings,                   # Global settings
    Document,                   # Document object
)
from llama_index.core.node_parser import (
    SentenceSplitter,          # Semantic chunking
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,   # Filter by similarity score
    MetadataReplacementPostProcessor,  # Replace node content with metadata
)
from llama_index.core.response_synthesizers import get_response_synthesizer

# LLM and Embedding providers
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Vector store (using Qdrant as example)
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client


# ============================================================================
# STEP 1: Configuration and Setup
# ============================================================================

def setup_environment():
    """
    Configure API keys and global settings.
    In production, use environment variables or secret managers.
    """
    # Set OpenAI API key
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Configure global LLM settings
    Settings.llm = OpenAI(
        model="gpt-4",
        temperature=0.1,  # Low temperature for factual responses
        max_tokens=1024,
    )
    
    # Configure global embedding model
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",  # Fast and cost-effective
        embed_batch_size=100,            # Batch embeddings for efficiency
    )
    
    # Configure chunking strategy
    Settings.node_parser = SentenceSplitter(
        chunk_size=512,        # Tokens per chunk
        chunk_overlap=50,      # Overlap to preserve context
        separator=" ",         # Split on spaces
    )
    
    print("✓ Environment configured")


# ============================================================================
# STEP 2: Document Ingestion and Processing
# ============================================================================

def load_documents(data_dir: str) -> List[Document]:
    """
    Load documents from a directory.
    Supports: PDF, TXT, DOCX, MD, HTML, JSON, CSV
    
    Args:
        data_dir: Path to directory containing documents
        
    Returns:
        List of Document objects with metadata
    """
    # SimpleDirectoryReader automatically detects file types
    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        recursive=True,              # Scan subdirectories
        required_exts=[".pdf", ".txt", ".md"],  # Filter file types
        filename_as_id=True,         # Use filename as document ID
    )
    
    documents = reader.load_data()
    
    # Enrich metadata (critical for filtering and citations)
    for doc in documents:
        doc.metadata.update({
            "source": doc.metadata.get("file_name", "unknown"),
            "doc_type": Path(doc.metadata.get("file_path", "")).suffix,
            "indexed_at": "2024-01-15",  # Add timestamp
        })
    
    print(f"✓ Loaded {len(documents)} documents")
    return documents


def create_custom_chunks(documents: List[Document]) -> List[Document]:
    """
    Advanced chunking with custom logic.
    Use when default SentenceSplitter is insufficient.
    
    Args:
        documents: Raw documents
        
    Returns:
        Chunked documents (nodes)
    """
    parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    
    # Parse documents into nodes (chunks)
    nodes = parser.get_nodes_from_documents(documents)
    
    # Add custom metadata to each chunk
    for i, node in enumerate(nodes):
        node.metadata["chunk_id"] = i
        node.metadata["chunk_size"] = len(node.text)
    
    print(f"✓ Created {len(nodes)} chunks")
    return nodes


# ============================================================================
# STEP 3: Vector Store Setup (Qdrant Example)
# ============================================================================

def setup_vector_store(collection_name: str = "rag_demo"):
    """
    Initialize Qdrant vector database.
    Can be replaced with FAISS, Pinecone, Weaviate, etc.
    
    Args:
        collection_name: Name of the vector collection
        
    Returns:
        QdrantVectorStore instance
    """
    # Connect to Qdrant (local or cloud)
    client = qdrant_client.QdrantClient(
        # Local mode
        path="./qdrant_data",
        
        # Cloud mode (uncomment and configure)
        # url="https://your-cluster.qdrant.io",
        # api_key="your-api-key",
    )
    
    # Create vector store wrapper
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )
    
    print(f"✓ Vector store '{collection_name}' ready")
    return vector_store


# ============================================================================
# STEP 4: Index Creation
# ============================================================================

def create_index(
    documents: List[Document],
    vector_store: Optional[QdrantVectorStore] = None,
) -> VectorStoreIndex:
    """
    Create a vector index from documents.
    
    Args:
        documents: Documents to index
        vector_store: Optional external vector store
        
    Returns:
        VectorStoreIndex ready for querying
    """
    if vector_store:
        # Use external vector store
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,  # Display progress bar
        )
    else:
        # Use in-memory vector store (default)
        index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True,
        )
    
    print("✓ Index created and populated")
    return index


# ============================================================================
# STEP 5: Advanced Retrieval Configuration
# ============================================================================

def create_advanced_retriever(index: VectorStoreIndex):
    """
    Configure retriever with advanced settings.
    
    Args:
        index: Vector index
        
    Returns:
        Configured retriever
    """
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,  # Retrieve top 10 candidates
        # Optional: add metadata filters
        # filters=MetadataFilters(
        #     filters=[
        #         MetadataFilter(key="doc_type", value=".pdf"),
        #     ]
        # ),
    )
    
    return retriever


def create_query_engine(index: VectorStoreIndex) -> RetrieverQueryEngine:
    """
    Create a query engine with retrieval and post-processing.
    
    Args:
        index: Vector index
        
    Returns:
        Query engine ready for questions
    """
    # Step 1: Configure retriever
    retriever = create_advanced_retriever(index)
    
    # Step 2: Configure post-processors (reranking, filtering)
    postprocessors = [
        # Filter out low-similarity results
        SimilarityPostprocessor(similarity_cutoff=0.7),
        
        # Optional: replace node content with metadata
        # MetadataReplacementPostProcessor(target_metadata_key="summary"),
    ]
    
    # Step 3: Configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="compact",  # Options: compact, tree_summarize, refine
        use_async=False,
    )
    
    # Step 4: Assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=postprocessors,
        response_synthesizer=response_synthesizer,
    )
    
    print("✓ Query engine configured")
    return query_engine


# ============================================================================
# STEP 6: Query Execution with Citations
# ============================================================================

def query_with_sources(query_engine: RetrieverQueryEngine, question: str):
    """
    Execute a query and return answer with source citations.
    
    Args:
        query_engine: Configured query engine
        question: User question
        
    Returns:
        Response object with answer and sources
    """
    # Execute query
    response = query_engine.query(question)
    
    # Extract answer
    print("\n" + "="*60)
    print("QUESTION:", question)
    print("="*60)
    print("\nANSWER:")
    print(response.response)
    
    # Extract and display sources
    print("\n" + "-"*60)
    print("SOURCES:")
    print("-"*60)
    
    for i, node in enumerate(response.source_nodes, 1):
        print(f"\n[{i}] Score: {node.score:.3f}")
        print(f"    Source: {node.metadata.get('source', 'unknown')}")
        print(f"    Excerpt: {node.text[:200]}...")
    
    return response


# ============================================================================
# STEP 7: Streaming Responses (Optional)
# ============================================================================

def query_with_streaming(query_engine: RetrieverQueryEngine, question: str):
    """
    Stream response tokens in real-time.
    Useful for user-facing applications.
    
    Args:
        query_engine: Configured query engine
        question: User question
    """
    streaming_response = query_engine.query(question)
    
    print("\n" + "="*60)
    print("STREAMING ANSWER:")
    print("="*60 + "\n")
    
    # Stream tokens as they arrive
    for text in streaming_response.response_gen:
        print(text, end="", flush=True)
    
    print("\n")


# ============================================================================
# STEP 8: Persistence (Save/Load Index)
# ============================================================================

def save_index(index: VectorStoreIndex, persist_dir: str = "./storage"):
    """
    Persist index to disk for reuse.
    
    Args:
        index: Index to save
        persist_dir: Directory path
    """
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"✓ Index saved to {persist_dir}")


def load_index(persist_dir: str = "./storage") -> VectorStoreIndex:
    """
    Load a previously saved index.
    
    Args:
        persist_dir: Directory containing saved index
        
    Returns:
        Loaded index
    """
    from llama_index.core import load_index_from_storage
    
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    
    print(f"✓ Index loaded from {persist_dir}")
    return index


# ============================================================================
# MAIN EXECUTION FLOW
# ============================================================================

def main():
    """
    Complete RAG pipeline execution.
    """
    # Step 1: Setup
    setup_environment()
    
    # Step 2: Load documents
    documents = load_documents("./data")  # Replace with your data directory
    
    # Step 3: Optional - Setup external vector store
    # vector_store = setup_vector_store("my_collection")
    vector_store = None  # Use in-memory for this example
    
    # Step 4: Create index
    index = create_index(documents, vector_store)
    
    # Step 5: Optional - Save index for reuse
    save_index(index, "./storage")
    
    # Step 6: Create query engine
    query_engine = create_query_engine(index)
    
    # Step 7: Execute queries
    questions = [
        "What are the main features of the product?",
        "How do I configure authentication?",
        "What are the pricing tiers?",
    ]
    
    for question in questions:
        query_with_sources(query_engine, question)
        print("\n" + "="*60 + "\n")


# ============================================================================
# ADVANCED: Custom Prompt Templates
# ============================================================================

def create_custom_prompt_query_engine(index: VectorStoreIndex):
    """
    Use custom prompts to control LLM behavior.
    """
    from llama_index.core import PromptTemplate
    
    # Define custom prompt
    qa_prompt = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query. If the answer is not in the context, "
        "respond with 'Insufficient information'.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    
    query_engine = index.as_query_engine(
        text_qa_template=qa_prompt,
        similarity_top_k=5,
    )
    
    return query_engine


# ============================================================================
# ADVANCED: Metadata Filtering
# ============================================================================

def query_with_filters(index: VectorStoreIndex, question: str):
    """
    Query with metadata filters (e.g., date range, document type).
    """
    from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
    
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="doc_type", value=".pdf"),
            # MetadataFilter(key="indexed_at", value="2024-01-15", operator=">="),
        ]
    )
    
    query_engine = index.as_query_engine(
        filters=filters,
        similarity_top_k=5,
    )
    
    response = query_engine.query(question)
    return response


if __name__ == "__main__":
    main()
