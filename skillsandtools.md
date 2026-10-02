# Skills and Tools

This document lists every skill and tool in the Enterprise AI Knowledge
Assistant, what each one does, and how the agent decides which one to use.

---

## What's a skill vs. a tool?

- **Skills** reason over retrieved knowledge. They combine information from
  the vector database with LLM reasoning to produce an answer.
- **Tools** are narrow, mostly deterministic operations invoked on demand.
  They don't reason about the knowledge base — they compute, parse, or
  transform whatever input they're given.

---

## Skills

### 1. Knowledge Retrieval
**File:** `skills/knowledge_retrieval.py`

Pure retrieval — no generation. Queries ChromaDB for the most relevant
knowledge-base chunks and returns them as-is, with their source document
names. Used directly by the `summarize` route to gather content before
handing it to the summarization tool, and available independently for any
future skill that needs raw retrieved context.

### 2. Policy Analysis
**File:** `skills/policy_analysis.py`

Retrieves policy-relevant chunks from the knowledge base, then uses
JSON-constrained LLM generation to produce a structured three-part answer:
**Policy information**, **Analysis**, and **Practical implication**. JSON
mode (`format="json"`) was adopted specifically to stop the model from
leaking raw chain-of-thought reasoning into the final answer, which was a
recurring failure mode with free-text generation on this model.

---

## Tools

### 1. Calculator
**File:** `tools/calculator/calculator.py`

Safely evaluates arithmetic expressions. Parses the expression into a
Python AST and walks it node-by-node, allowing only numeric constants and
the operators `+ - * /` (plus unary +/-) — never calling Python's `eval()`
directly, so there's no path to arbitrary code execution. Rejects anything
else (imports, names, function calls) with a clear error.

The agent's `extract_math_expression()` function feeds this tool a plain
expression after converting natural language into one — either via fast,
deterministic regex patterns (percentages, "X out of Y", value changes,
monthly-to-annual conversion, etc.) or, for less common phrasing, via an
LLM few-shot prompt that translates the question into an expression
without solving it itself. The LLM's output is validated the same way
before it ever reaches the calculator.

### 2. Date/Time
**File:** `tools/date_time/date_time.py`

Answers date and time questions:
- Today's date
- Days until / since a given date (ISO format or month name)
- Days between two dates
- A date N days from today

### 3. Document Summarization
**File:** `tools/document_summarization/document_summarization.py`

Summarizes a block of text passed to it. Does not retrieve anything
itself — the agent's `summarize` route first calls the Knowledge Retrieval
skill to gather relevant chunks, concatenates them, and passes the result
into this tool. This two-step chain (retrieve, then summarize) is the
project's clearest example of multi-step agentic execution.

---

## Supporting infrastructure

| Component | File | Role |
|---|---|---|
| Knowledge Q&A | `rag/ask.py` | Retrieval + JSON-mode grounded generation for general factual questions |
| Shared retrieval | `rag/retrieve.py` | The ChromaDB query function used by both `rag/ask.py` and `skills/knowledge_retrieval.py` |
| Agent orchestrator | `agent/agent.py` | Classifies each question's intent and routes it to the correct skill or tool |

---

## How routing works

The agent (`classify_intent()` in `agent/agent.py`) checks, in order:

1. **Fast-path date detection** — regex for ISO dates, month names, and
   date-related keywords ("today", "days until", etc.). No LLM call.
2. **Fast-path calculator detection** — regex for math symbols, two or
   more numbers, or number-words paired with calculation language. No
   LLM call.
3. **LLM classification** — for everything else, a prompt asks the model
   to choose between `knowledge_qa`, `policy_analysis`, `summarize`, or
   `out_of_scope`. The response is parsed using explicit `ANSWER_START`/
   `ANSWER_END` markers rather than scanning raw model output, since the
   model's own reasoning text can otherwise contain misleading false
   matches.
4. **Keyword fallback** — if the LLM call fails or returns something
   unparseable, simple keyword matching decides between `summarize`,
   `policy_analysis`, and `knowledge_qa` (the default).

This fast-path-first design means the two most common question types —
math and dates — never depend on the LLM at all, which is both faster and
more reliable than routing everything through a single classification
call.