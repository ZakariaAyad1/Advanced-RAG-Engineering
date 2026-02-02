from typing import TypedDict, List, Annotated, Union
from dataclasses import dataclass
import operator

# =============================================================================
# I. THE ESSENCE OF STATE SCHEMAS (TypedDict & Dataclass)
# =============================================================================
# TypedDict provides a way to specify the keys and value types of a dictionary.
# It is the most common way to define state in LangGraph due to its simplicity.

class AgentState(TypedDict):
    """
    Represents the internal state of a LangGraph agent using TypedDict.
    It allows static type checkers (like mypy) to ensure key safety.
    Note: These are type hints and are not enforced at runtime.
    """
    query: str
    chat_history: List[str]
    # 'Annotated' allows us to attach metadata or 'reducers' to the state.
    # In LangGraph, operator.add tells the graph to append new results 
    # rather than overwriting the previous list.
    documents: Annotated[List[str], operator.add]
    next_step: str

# -----------------------------------------------------------------------------
# ALTERNATIVE: DataClass State Schema
# -----------------------------------------------------------------------------
# As seen in your notebook (3-DataclassStateSchema.ipynb), we can also use 
# Python's @dataclass. This is useful when you want an object-oriented 
# approach or need to define default values more naturally.

@dataclass
class DataClassState:
    """
    An alternative state schema using @dataclass.
    LangGraph supports this for developers who prefer dot-notation 
    (state.name) over key-access (state['name']).
    """
    name: str
    game: str
    # Example of a reducer in a dataclass-based state:
    # notes: Annotated[list, operator.add]

# =============================================================================
# II. STATE MANAGEMENT COMPARISON
# =============================================================================
# A critical architectural decision in AI Engineering is choosing the 
# right tool for the specific layer of the application.

# 1. TypedDict:
#    - Access: state["key"]
#    - Use for: Standard LangGraph states, lightweight flows.
#    - Benefit: Most compatible with existing LangChain components.

# 2. DataClass:
#    - Access: state.key
#    - Use for: Complex logic where object-oriented patterns are preferred.
#    - Benefit: Cleaner syntax for attribute access and default factory values.

# 3. Pydantic (BaseModel):
#    - Use for: External interfaces, LLM Structured Output.
#    - Feature: Runtime validation (it throws an error if data is wrong).

# =============================================================================
# III. PRACTICAL IMPLEMENTATION IN AGENTIC LOOPS
# =============================================================================

def research_agent_node(state: AgentState) -> dict:
    """
    A simulated node in a LangGraph workflow.
    Notice the function signature uses the TypedDict for clarity.
    """
    current_query = state["query"]
    print(f"--- Processing Query: {current_query} ---")
    
    # Simulating a tool call or retrieval
    new_evidence = ["Semantic search utilizes cosine similarity.", "MMR ensures diversity."]
    
    # We return a dictionary that updates the state.
    # Because of the 'Annotated' reducer in the AgentState definition,
    # these documents will be appended to the existing list.
    return {
        "documents": new_evidence,
        "next_step": "summarize"
    }

# --- Execution ---
initial_state: AgentState = {
    "query": "How does vector retrieval work?",
    "chat_history": [],
    "documents": [],
    "next_step": "start"
}

# Update state via node logic
update = research_agent_node(initial_state)

# In a real LangGraph environment, the 'Graph' object manages this merge.
# Here we simulate the result:
final_state = {**initial_state, **update}
print(f"\nUpdated Agent State: {final_state}")

# =============================================================================
# SUMMARY FOR THE AGENTIC ARCHITECT:
# - Use TypedDict to define the 'Shape' of your Agent's memory (Standard).
# - Use DataClasses for a cleaner, object-oriented state access (Alternative).
# - Use Annotated to define 'Reducers' (how data merges over time).
# =============================================================================
