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
Agent (fast-path routing + LLM classification)
     |
     +-------------+-------------+-------------+-------------+
     |             |             |             |             |
     v             v             v             v             v
Knowledge      Policy        Calculator    Date/Time     Summarize
Retrieval      Analysis      (regex +      (regex only)  (retrieval +
(RAG + JSON)   (RAG + JSON)   AST + LLM)                 summarizer)
     |             |             |             |             |
     +-------------+-------------+-------------+-------------+
                             |
                             v
                    ChromaDB + Ollama
                             |
                             v
                         Response
                   (answer + sources + route)
                             |
                             v
                     Streamlit UI
```

A question with no clear match for either knowledge or capability falls
through to an `out_of_scope` response rather than being forced into one of
the five routes above.

The agent (`agent/agent.py`) acts as the orchestrator, in two stages:

1. **Fast-path detection (no LLM call).** Regex checks first identify
   obvious date questions and obvious math questions. This means the two
   most common question types never depend on the LLM's judgment at all —
   both faster and more reliable than routing everything through a single
   classification call.
2. **LLM classification (for everything else).** A prompt asks the model
   to choose between `knowledge_qa`, `policy_analysis`, `summarize`, or
   `out_of_scope`. The response is parsed using explicit `ANSWER_START` /
   `ANSWER_END` markers rather than scanning the model's raw output —
   scanning raw text for category names is unreliable, since a model's own
   reasoning text often mentions every category while "thinking out loud,"
   producing false matches. If the LLM call fails or returns something
   unparseable, a keyword fallback decides between `summarize`,
   `policy_analysis`, and `knowledge_qa` (the default).

---

## Skills vs. Tools

These terms are used deliberately throughout the codebase.

| | Skills | Tools |
|---|---|---|
| **What they are** | Capabilities that reason over or analyze knowledge | Operations invoked by the agent for specific tasks |
| **Where they live** | `skills/` | `tools/` |
| **In this project** | `knowledge_retrieval.py`, `policy_analysis.py` | `calculator/`, `date_time/`, `document_summarization/` |
| **Typical implementation** | Retrieval + LLM reasoning | Deterministic logic and/or LLM-assisted processing |

`document_summarization` is implemented as a tool. The `summarize` route
retrieves relevant document content via the Knowledge Retrieval skill and
then invokes the summarization tool, demonstrating combined retrieval and
tool usage.

---

## Capabilities Demonstrated

| # | Capability | Where it is demonstrated |
|---|---|---|
| 1 | Knowledge retrieval | `knowledge_qa` route using ChromaDB + LLM |
| 2 | Skill selection | Fast-path routing and LLM intent classification in `agent.py` |
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
│   └── agent.py                 # Fast-path + LLM intent classification, routing
│
├── skills/
│   ├── knowledge_retrieval.py   # Pure retrieval (no generation)
│   └── policy_analysis.py       # Retrieval + structured JSON-mode LLM analysis
│
├── tools/
│   ├── calculator/               # AST-validated arithmetic evaluation
│   ├── date_time/                 # ISO + month-name date parsing
│   └── document_summarization/   # Summarizes text passed to it
│
├── rag/
│   ├── ask.py                   # Knowledge Q&A (retrieval + JSON-mode generation)
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
- **Regex / deterministic logic** — fast handling of structured calculator, date, and routing cases
- **AST-based expression validation** — the calculator never calls Python's `eval()` directly; expressions are parsed into an AST and walked node-by-node, allowing only numeric constants and arithmetic operators

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

The calculator pipeline separates **expression extraction** from **expression evaluation**, and tries the fastest, most reliable method first:

```text
Natural-language question
        |
        v
Already a plain expression? ----yes----> skip ahead to validation
        | no
        v
Deterministic regex patterns
(percent-of, increase/decrease, value
 changes, monthly-to-annual, etc.)
        |
        | no match
        v
LLM few-shot translation
(converts meaning into an expression,
 does not solve it itself)
        |
        v
AST-based validation
(numeric constants + arithmetic
 operators only — no eval() of
 arbitrary input)
        |
        v
Calculator evaluates
        |
        v
Result
```

Most common phrasings (basic arithmetic, percentages, increases/decreases,
monthly-to-annual conversion) are resolved by the deterministic regex
patterns alone, with no LLM call needed. Only less common or more
open-ended phrasing falls through to the LLM translation step — which
still produces a plain expression, validated the same way, before the
calculator ever evaluates it.

When required information is missing, the system is designed to avoid
inventing a value and instead indicate that the calculation cannot be
completed from the provided information.

---

## Policy Analysis

The policy analysis capability retrieves relevant information from the governance knowledge base and uses the local LLM, constrained to JSON output, to produce a structured three-part answer: **policy information**, **analysis**, and **practical implication**.

JSON-constrained generation (`format="json"` in the Ollama API call) was adopted specifically after free-text generation repeatedly let the model's internal reasoning leak into the final answer — this local model tends to "think out loud" in plain text even when explicitly instructed not to, and constraining the output grammar from the first generated token proved far more reliable than trying to strip reasoning out afterward.

This allows the system to distinguish between:

- Information supported by the governance policy
- Information available in the broader enterprise AI knowledge base
- Questions for which the knowledge base does not contain sufficient information

---

## Knowledge Q&A

General factual questions about enterprise AI (capabilities, risks,
implementation approach, success metrics) are answered the same way as
policy questions — retrieval followed by JSON-constrained generation —
for the same reason: it keeps the model's raw reasoning out of the final
answer shown to the user.

---

## Date / Time

The date/time tool resolves entirely through regex, with no LLM call at
any point. It supports:

- Today's date, current year
- A date in `YYYY-MM-DD` format, or by month name (e.g. "December 31")
- Days until / since a given date
- Days between two dates
- A date N days from today

---

## Document Summarization

The document summarization capability retrieves relevant document content via the Knowledge Retrieval skill and generates a concise summary based on the retrieved information.

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

Routes that combine retrieval with LLM reasoning (knowledge Q&A, policy analysis) are noticeably slower than routes that resolve without any LLM call (date/time, and most calculator questions).

### Reasoning leakage, and why JSON mode is used

Qwen3 4B is a hybrid-reasoning model: even with `think=False` set and explicit "do not show your reasoning" instructions in the prompt, it would frequently reason in plain text before producing its real answer, and that reasoning could leak into the final output if the model ran out of its token budget before finishing. JSON-constrained generation (`format="json"`) proved to be a more reliable fix than trying to detect and strip reasoning after the fact, since it constrains the output grammar from the very first generated token.

### Hardware

The project was developed and tested using a MacBook Air with an M2 processor and 8 GB RAM.

Qwen3 4B was selected as a practical local-model choice for this hardware. Larger models may require substantially more memory and may increase response latency.

### Calculator extraction

The calculator handles common mathematical and business-calculation patterns directly via deterministic regex, and uses LLM-assisted extraction for more varied natural-language phrasing. Every extracted expression — whether from regex or the LLM — passes through the same AST-based validation before being evaluated.

Because natural-language interpretation for uncommon phrasing is performed by a language model, unusual or ambiguous questions may still require additional validation.

The system is designed to avoid inventing missing numerical information.

### Single-turn interaction

The current agent does not maintain conversational memory across turns.

Each question is classified and processed independently.

### Single-intent routing

The current agent is designed around **single-intent questions**. A question containing multiple independent tasks is routed according to the detected primary intent rather than executing multiple capabilities and combining the results.

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