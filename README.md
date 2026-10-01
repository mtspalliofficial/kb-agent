# Enterprise AI Knowledge Assistant

A small agentic AI application that understands a natural-language question, determines which capability it needs, retrieves relevant knowledge or invokes a tool, and returns a grounded response with sources where applicable.

Built as a 2-day final-year Computer Science project demonstrating practical integration of an agent, skills, tools, a vector database, an LLM, and a Streamlit interface.

---

## Architecture

See `architecture_diagram.svg` for the full visual flow.

```text
User query
     |
     v
Streamlit UI
     |
     v
Agent (intent classification + routing)
     |
     +-------------------+-------------------+-------------------+
     |                   |                   |                   |
     v                   v                   v                   v
Knowledge           Policy Analysis      Calculator       Document
Retrieval (RAG)         (RAG + LLM)      (LLM + logic)    Summarization
     |                   |                   |                   |
     +-------------------+-------------------+-------------------+
                             |
                             v
                    ChromaDB + Ollama
                             |
                             v
                         Response
                   (answer + sources)
                             |
                             v
                     Streamlit UI
```

The agent (`agent/agent.py`) acts as the orchestrator. It classifies the user's intent using a local LLM and routes the question to the appropriate capability.

Because a small local 4B model can occasionally produce unexpected classification output, the agent validates the model's classification and uses deterministic fallback routing when the expected output is not produced. This hybrid approach improves routing reliability while keeping the system local.

---

## Skills vs. Tools

These terms are used deliberately throughout the codebase.

| | Skills | Tools |
|---|---|---|
| **What they are** | Capabilities that reason over or analyze knowledge | Operations invoked by the agent for specific tasks |
| **Where they live** | `skills/` | `tools/` |
| **In this project** | `knowledge_retrieval.py`, `policy_analysis.py` | `calculator/`, `date_time/`, `document_summarization/` |
| **Typical implementation** | Retrieval + LLM reasoning | Deterministic logic and/or LLM-assisted processing |

`document_summarization` is implemented as a tool. The `summarize` route retrieves relevant document content and then invokes the summarization capability, demonstrating combined retrieval and tool usage.

---

## Capabilities Demonstrated

| # | Capability | Where it is demonstrated |
|---|---|---|
| 1 | Knowledge retrieval | `knowledge_qa` route using ChromaDB + LLM |
| 2 | Skill selection | Intent classification and routing in `agent.py` |
| 3 | Tool invocation | Calculator, date/time, and document summarization routes |
| 4 | Combined retrieval + tool usage | Summarization route retrieves relevant content before summarizing |
| 5 | Source-grounded responses | Routes expose retrieved source information where applicable |
| 6 | Multi-step agentic execution | `run_agent()` classifies, routes, retrieves/processes information, and returns the result |
| 7 | Unsupported-question handling | Out-of-scope routing and grounded "not enough information" responses |
| 8 | Natural-language calculation | Calculator converts varied natural-language questions into mathematical expressions before evaluation |

---

## Project Structure

```text
kb-agent/
│
├── agent/
│   └── agent.py                 # Intent classification + routing
│
├── skills/
│   ├── knowledge_retrieval.py   # Knowledge retrieval
│   └── policy_analysis.py       # Retrieval + structured LLM analysis
│
├── tools/
│   ├── calculator/
│   ├── date_time/
│   └── document_summarization/
│
├── rag/
│   ├── ask.py                   # Knowledge Q&A
│   ├── retrieve.py              # Shared ChromaDB retrieval
│   └── build_index.py           # Builds/rebuilds the ChromaDB index
│
├── knowledge_base/
│   ├── enterprise_ai_overview.txt
│   └── ai_governance_policy.txt
│
├── tests/
│   ├── knowledge_retrieval/
│   ├── policy_analysis/
│   ├── document_summarization/
│   ├── calculator/
│   ├── agent_routing/
│   └── integration/
│
├── app.py                       # Streamlit UI
├── requirements.txt
└── README.md
```

---

## Knowledge Base

The project currently uses two knowledge-base documents:

### `enterprise_ai_overview.txt`

Contains information about enterprise AI capabilities, risks, implementation approaches, digital transformation, and AI project success metrics.

It is intentionally not a policy document. Therefore, policy-specific questions that are not supported by this document should not be answered by inventing information.

### `ai_governance_policy.txt`

Contains concrete AI governance policy information covering areas such as data privacy, human oversight, model risk, security, compliance, and vendor review.

This document provides the policy-specific knowledge used by the policy analysis capability.

---

## Core Technologies

- **Python** — application and agent logic
- **Qwen3 4B** — local language model
- **Ollama** — local LLM runtime
- **ChromaDB** — vector database
- **nomic-embed-text** — embedding model
- **Streamlit** — user interface
- **Regex / deterministic logic** — fast handling of structured calculator and routing cases

---

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Pull the required Ollama models:

```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

Build the vector database:

```bash
python rag/build_index.py
```

Launch the application:

```bash
streamlit run app.py
```

The application requires Ollama to be installed and running locally.

---

## Calculator

The calculator supports both direct arithmetic and natural-language calculation requests.

Examples include:

```text
What is 133 * 165?

What is 15% of 2000?

If an AI system saves 25 hours per week at $800 per hour,
what are the annual savings?

An AI implementation costs $120,000 and saves $10,000
per month. How many months until break-even?
```

The calculator pipeline separates **expression extraction** from **expression evaluation**:

```text
Natural-language question
        ↓
Expression extraction
        ↓
Validated mathematical expression
        ↓
Calculator
        ↓
Result
```

The calculator does not directly execute arbitrary natural-language input. The extracted expression is restricted to supported mathematical operations before evaluation.

When required information is missing, the system is designed to avoid inventing a value and instead indicate that the calculation cannot be completed from the provided information.

---

## Policy Analysis

The policy analysis capability retrieves relevant information from the governance knowledge base and uses the local LLM to produce a structured answer.

It is grounded in the project's knowledge base rather than relying on unsupported external information.

This allows the system to distinguish between:

- Information supported by the governance policy
- Information available in the broader enterprise AI knowledge base
- Questions for which the knowledge base does not contain sufficient information

---

## Document Summarization

The document summarization capability retrieves relevant document content and generates a concise summary based on the retrieved information.

It supports requests such as:

```text
Summarize the document.

Give me the key points.

Summarize only the risks.

What are the main recommendations?

Give me an executive summary.
```

---

## Source Grounding

Where retrieval is involved, the application exposes the relevant source information alongside the response.

This allows users to see which knowledge-base content was used to generate the answer and helps reduce unsupported responses.

For questions outside the available knowledge base, the system can return a grounded response indicating that there is insufficient information rather than fabricating an answer.

---

## Testing

The project includes categorized test scenarios under `tests/`.

```text
tests/
├── knowledge_retrieval/
│   ├── basic_questions/
│   ├── contextual_questions/
│   └── unknown_questions/
│
├── policy_analysis/
│   ├── governance/
│   ├── risk_analysis/
│   └── responsible_ai/
│
├── document_summarization/
│   ├── short_documents/
│   ├── long_documents/
│   └── targeted_summaries/
│
├── calculator/
│   ├── basic_arithmetic/
│   ├── percentages/
│   ├── natural_language/
│   ├── multi_step/
│   └── edge_cases/
│
├── agent_routing/
│   └── single_intent/
│
└── integration/
    ├── end_to_end/
    └── error_handling/
```

The test scenarios cover:

- Basic knowledge questions
- Contextual retrieval
- Unknown/out-of-scope questions
- Governance and risk analysis
- Responsible AI
- Document summarization
- Natural-language calculations
- Multi-step calculations
- Calculator edge cases
- Agent routing
- End-to-end execution
- Error handling

---

## Known Limitations

### Local LLM latency

The application runs the LLM locally through Ollama. Response time therefore depends on the hardware running the model.

Some routes require more processing than others, particularly routes that combine retrieval with additional LLM reasoning.

### Hardware

The project was developed and tested using a MacBook Air with an M2 processor and 8 GB RAM.

Qwen3 4B was selected as a practical local-model choice for this hardware. Larger models may require substantially more memory and may increase response latency.

### Calculator extraction

The calculator handles common mathematical and business-calculation patterns directly and uses LLM-assisted extraction for more varied natural-language phrasing.

Because natural-language interpretation is performed by a language model, unusual or ambiguous questions may still require additional validation.

The system is designed to avoid inventing missing numerical information.

### Single-turn interaction

The current agent does not maintain conversational memory across turns.

Each question is classified and processed independently.

### Single-intent routing

The current agent is designed around **single-intent questions**. A question containing multiple independent tasks may be routed according to the detected primary intent rather than executing multiple capabilities simultaneously.

For example:

```text
What are the risks of enterprise AI, and what is 15% of 20,000?
```

contains both a knowledge request and a calculation. The current architecture is not designed to combine both results into one response.

---

## Future Improvements

Potential future improvements include:

- Multi-intent question handling
- Conversation memory
- More advanced calculator validation
- Additional enterprise knowledge sources
- Authentication and access control
- Persistent production storage
- Cloud deployment
- Monitoring and logging
- Automated test execution
- Evaluation metrics for retrieval and routing accuracy

---

## Project Summary

The Enterprise AI Knowledge Assistant demonstrates how a small local language model can be combined with retrieval, specialized capabilities, deterministic tools, and a user interface to create a practical agentic AI application.

The project focuses on **grounded responses, capability-based routing, tool usage, local execution, and safe handling of unsupported questions** rather than simply generating unrestricted LLM responses.