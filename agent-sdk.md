An **Agent SDK (Software Development Kit)** is a set of tools, libraries, and APIs designed to help developers build **AI Agents**.

While a standard LLM (like a basic chatbot) simply predicts the next word in a sentence, an **Agent** built with an SDK can reason, use tools, and interact with the real world to complete complex, multi-step goals.

---

### The Core Components of an Agent SDK

Most SDKs (like the one you referenced earlier) provide a framework to manage four main "pillars" of agency:

#### 1. Perception & Reasoning (The Brain)

The SDK provides the interface to the LLM. It manages the **System Instructions** (the "who am I" part) and the **Reasoning Loop** (often using patterns like ReAct: Reason + Act).

* **Example:** The SDK helps the agent decide: "To answer this question, I first need to check the user's database."

#### 2. Tool Use / Function Calling (The Hands)

This is the most critical part of an SDK. It allows you to "plug in" external capabilities.

* **APIs:** Connecting to Slack, GitHub, or Google Calendar.
* **Custom Code:** Using a `function_tool` to let the agent run your specific Python scripts.
* **Search:** Giving the agent a `WebSearchTool` to find current info.

#### 3. Memory (The Context)

SDKs manage how the agent remembers past interactions.

* **Short-term:** The current conversation history.
* **Long-term:** Connecting to "Vector Databases" (RAG) so the agent can look up documents or facts from weeks ago.

#### 4. Orchestration (The Manager)

The **Runner** or **Orchestrator** within the SDK handles the "loop." It sends a prompt, gets a tool request back, executes that tool, and feeds the result back to the LLM until the job is done.

---

### Why use an SDK instead of just an API?

If you use a raw LLM API, you have to manually write the code for every loop, every error, and every tool connection. An SDK handles the "boilerplate" code for you:

| Feature | Raw LLM API | Agent SDK |
| --- | --- | --- |
| **State Management** | You must save/load history manually. | Handled automatically. |
| **Tool Execution** | You must parse JSON and call functions yourself. | The **Runner** executes tools for you. |
| **Error Handling** | If the AI hallucinates a tool name, the code crashes. | The SDK often has "self-correction" loops. |
| **Tracing** | Hard to see why the AI made a mistake. | Built-in **tracing** logs every thought and action. |

---

### Popular Agent SDKs today

* **LangChain / LangGraph:** Highly modular and widely used.
* **CrewAI:** Focused on "Multi-Agent" orchestration (making agents talk to each other).
* **AutoGPT/OpenAI Assistants:** High-level abstractions for autonomous tasks.
* **Vertex AI Agent SDK:** Google’s enterprise-grade framework for building and deploying agents.

**Would you like to see a "Hello World" example of an agent using an SDK to perform a specific task, like checking the weather or searching the web?**
