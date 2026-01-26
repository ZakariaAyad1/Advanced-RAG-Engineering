import json
from typing import List, Optional, Union, Dict
from pydantic import BaseModel, Field, validator
from datetime import datetime

# =============================================================================
# I. FUNDAMENTAL STATIC TYPING (Type Hinting)
# =============================================================================
# In modern Python (3.9+), we use type hints to declare the expected data 
# types of variables and function signatures. This facilitates static 
# analysis and significantly enhances code legibility in complex RAG flows.

def process_document_metadata(
    doc_id: str, 
    word_count: int, 
    tags: List[str], 
    score: Optional[float] = None
) -> Dict[str, Union[str, int, float, List[str]]]:
    """
    Demonstrates standard type hinting for a metadata processor.
    - doc_id: Must be a string.
    - tags: Must be a list of strings.
    - score: Can be a float or None (Optional).
    """
    return {
        "id": doc_id,
        "length": word_count,
        "keywords": tags,
        "relevance": score or 0.0
    }


# =============================================================================
# II. PYDANTIC BASEMODEL: DATA VALIDATION & SCHEMA ENFORCEMENT
# =============================================================================
# While type hints are purely advisory, Pydantic's BaseModel enforces types 
# at runtime. This is the gold standard for "Structured Output" from LLMs.

class RAGChunk(BaseModel):
    """
    Represents a semantically chunked unit of text for a Vector Database.
    Inheriting from BaseModel provides automatic data validation and 
    serialization capabilities.
    """
    
    # 1. Attribute Declarations with Type Hints
    chunk_id: int
    content: str = Field(..., min_length=10, description="The textual content of the chunk.")
    
    # 2. Metadata with Defaults
    # Field allows for additional constraints and metadata documentation.
    source_uri: str
    embedding: List[float] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    
    # 3. Custom Validation Logic
    # Validators allow for complex logical constraints beyond simple types.
    @validator('chunk_id')
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('The chunk_id must be a positive integer.')
        return v

class RetrievalResponse(BaseModel):
    """
    A container model representing a multi-chunk retrieval response.
    Models can be nested to create complex, hierarchical data structures.
    """
    query: str
    results: List[RAGChunk]
    latency_ms: float


# =============================================================================
# III. PRACTICAL EXECUTION & SERIALIZATION
# =============================================================================

def run_demonstration():
    # --- Example 1: Successful Validation ---
    raw_data = {
        "chunk_id": 101,
        "content": "Maximal Marginal Relevance (MMR) optimizes for diversity.",
        "source_uri": "https://docs.langchain.com/mmr",
        "embedding": [0.12, -0.45, 0.88]
    }
    
    # Instantiate the model. Pydantic automatically converts types if possible.
    chunk = RAGChunk(**raw_data)
    print(f"✅ Successfully validated chunk: {chunk.chunk_id}")
    
    # --- Example 2: Automatic Serialization ---
    # Convert the object back to a Python Dict or JSON string effortlessly.
    json_output = chunk.json(indent=2)
    print(f"\nSerialized JSON Output:\n{json_output}")
    
    # --- Example 3: Handling Validation Errors ---
    print("\n⚠️ Attempting to instantiate invalid data...")
    invalid_data = {
        "chunk_id": -5,       # Triggers custom validator (must be positive)
        "content": "Short",   # Triggers Field constraint (min_length=10)
        "source_uri": "http://example.com"
    }
    
    try:
        RAGChunk(**invalid_data)
    except Exception as e:
        print(f"Caught Expected Validation Error:\n{e}")

if __name__ == "__main__":
    run_demonstration()

# =============================================================================
# SUMMARY OF BENEFITS:
# 1. Runtime Safety: Prevents "NoneType" or "AttributeError" deep in RAG chains.
# 2. IDE Support: Autocompletion works perfectly because types are known.
# 3. LLM Integration: Models like GPT-4 can be forced to return JSON that 
#    matches these Pydantic schemas exactly using "Function Calling".
# =============================================================================
