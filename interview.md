
# 🔥 1. How do you improve retrieval quality in a RAG system?

### Strong answer:

Improving retrieval quality is multi-dimensional:

* **Better chunking**

  * Semantic chunking instead of fixed size
  * Maintain context boundaries (sections, headings)

* **Hybrid retrieval**

  * Combine dense (embeddings) + sparse (BM25)

* **Re-ranking**

  * Use cross-encoders (e.g. BGE reranker) to rescore top-k results

* **Query rewriting**

  * Expand or clarify ambiguous queries

* **Metadata filtering**

  * Filter by source, time, category

* **Embedding selection**

  * Domain-specific embeddings if needed

👉 Key insight:

> Retrieval is not one step — it’s a pipeline.

---

# 🔥 2. Why does RAG still hallucinate?

### Strong answer:

RAG reduces hallucination but doesn’t eliminate it because:

* Retrieved context may be:

  * irrelevant
  * incomplete
  * contradictory

* LLM behavior:

  * fills gaps when context is insufficient
  * prioritizes fluency over factuality

* Prompt issues:

  * weak grounding instructions

👉 Fixes:

* stricter prompts (“answer ONLY from context”)
* confidence scoring
* answer verification step

👉 Brutal truth:

> If retrieval is weak, hallucination is inevitable.

---

# 🔥 3. What is the difference between dense, sparse, and hybrid retrieval?

### Strong answer:

* **Dense retrieval**

  * Uses embeddings
  * Captures semantic meaning

* **Sparse retrieval (BM25)**

  * Keyword-based
  * Strong for exact matches

* **Hybrid retrieval**

  * Combines both
  * Best in production

👉 Insight:

> Dense fails on exact keywords, sparse fails on semantics → hybrid fixes both.

---

# 🔥 4. How do you choose chunk size?

### Strong answer:

It’s a trade-off:

* Small chunks:

  * * precise retrieval
  * – lose context

* Large chunks:

  * * more context
  * – noisy retrieval

Best approach:

* 200–500 tokens baseline
* Add overlap (10–20%)
* Use **semantic chunking when possible**

👉 Bonus:
Evaluate using retrieval metrics, not guesswork.

---

# 🔥 5. What is re-ranking and why is it important?

### Strong answer:

Re-ranking is a second-stage scoring step:

1. Retrieve top-k candidates (fast, rough)
2. Re-score them with a more accurate model

Models:

* Cross-encoders
* LLM-based ranking

👉 Why it matters:

* Initial retrieval is noisy
* Re-ranking significantly improves precision

👉 Real insight:

> Most performance gains in RAG come from re-ranking, not embeddings.

---

# 🔥 6. How do you evaluate a RAG system?

### Strong answer:

You evaluate **both retrieval and generation**:

#### Retrieval:

* Recall@k
* Precision@k

#### Generation:

* Faithfulness (is it grounded?)
* Relevance
* Correctness

Tools:

* RAGAS
* DeepEval
* Human evaluation

👉 Insight:

> If you only evaluate final answers, you can’t debug failures.

---

# 🔥 7. What are common failure modes in RAG?

### Strong answer:

* Poor chunking
* Low-quality embeddings
* Wrong top-k
* Missing re-ranking
* Context overflow
* Latency issues

👉 Key insight:

> Most failures are retrieval failures, not LLM failures.

---

# 🔥 8. How do you reduce latency in RAG?

### Strong answer:

* Use ANN indexing (HNSW)
* Reduce top-k
* Cache embeddings and results
* Use smaller models for re-ranking
* Parallelize retrieval + generation

👉 Trade-off:
Speed vs accuracy

---

# 🔥 9. What is multi-query retrieval?

### Strong answer:

Instead of one query:

* Generate multiple reformulations
* Retrieve results for each
* Merge them

👉 Benefit:
Improves recall and coverage

---

# 🔥 10. What is context compression?

### Strong answer:

Reduce retrieved content before sending to LLM:

* Summarization
* Extract relevant sentences
* Remove redundancy

👉 Why:
LLM context window is limited and expensive

---

# 🔥 11. When would you NOT use RAG?

### Strong answer:

* Small static knowledge → just prompt
* High precision logic → use code
* Stable knowledge → fine-tune instead

---

# 🔥 12. RAG vs Fine-tuning?

### Strong answer:

* RAG → dynamic knowledge
* Fine-tuning → behavioral adaptation

👉 Best systems:
Use both together

---

# 🔥 13. What is HNSW and why is it used?

### Strong answer:

HNSW = graph-based ANN algorithm

* Builds multi-layer graph of vectors
* Enables fast nearest neighbor search

👉 Used because:

* high recall
* low latency
* scalable

---

# 🔥 14. How do you handle long documents?

### Strong answer:

* Chunking with overlap
* Hierarchical retrieval
* Summarization layers

---

# 🔥 15. How do you prevent irrelevant context from hurting answers?

### Strong answer:

* Re-ranking
* Context filtering
* Lower top-k
* Better chunking

👉 Insight:

> More context often makes answers worse, not better.

---

# 🔥 16. What is agentic RAG?

### Strong answer:

RAG where LLM:

* decides when to retrieve
* performs multi-step retrieval
* interacts with tools

---

# 🔥 17. What is query rewriting?

### Strong answer:

Transform user query into better search query:

* clarify ambiguity
* add context
* expand keywords

---

# 🔥 18. What are embeddings limitations?

### Strong answer:

* Miss exact keyword matches
* Domain sensitivity
* Expensive at scale

---

# 🔥 19. How do you handle conflicting documents?

### Strong answer:

* Source ranking
* Confidence scoring
* Return multiple viewpoints
* Add citation system

---

# 🔥 20. What is the biggest misconception about RAG?

### Strong answer:

> “RAG is just vector DB + LLM”

Reality:

* It’s a **retrieval system problem**
* LLM is just the final layer

---

