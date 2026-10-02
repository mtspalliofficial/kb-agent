# Technical Documentation
## Enterprise AI Knowledge Assistant

A short technical report accompanying the source code, architecture
diagram, and test scenarios submitted for this project.

---

## 1. Project Overview

The Enterprise AI Knowledge Assistant is an agentic AI application that
accepts a natural-language question, determines which capability is
needed to answer it, executes that capability (retrieval, a deterministic
tool, or a combination of both), and returns a grounded response with
source attribution where applicable.

The system runs entirely locally using Ollama (Qwen3 4B) as the language
model and ChromaDB as the vector database, requiring no external API
calls or internet access at inference time.

---

## 2. System Architecture

```text
User query -> Streamlit UI -> Agent -> one of five routes -> Response
```

The agent (`agent/agent.py`) classifies each question in two stages:

1. **Fast-path detection.** Regex rules check first for obvious date
   questions (ISO dates, month names, words like "today" or "days until")
   and obvious math questions (numeric symbols, two or more numbers,
   calculation keywords). Matched questions route immediately — no LLM
   call is made.
2. **LLM classification.** Anything that doesn't match a fast path is
   classified by the LLM into `knowledge_qa`, `policy_analysis`,
   `summarize`, or `out_of_scope`. The model is asked to wrap its answer
   in `ANSWER_START` / `ANSWER_END` markers; only text between those
   markers is trusted. A keyword-based fallback handles cases where the
   LLM call fails or returns an unparseable response.

Full diagram: `architecture_diagram.svg`.

---

## 3. Skills and Tools

| Type | Name | Purpose |
|---|---|---|
| Skill | Knowledge Retrieval | Pure vector-DB retrieval, no generation |
| Skill | Policy Analysis | Retrieval + structured JSON-mode LLM analysis |
| Tool | Calculator | AST-validated arithmetic evaluation |
| Tool | Date/Time | Regex-only date parsing and arithmetic |
| Tool | Document Summarization | Summarizes text handed to it by the agent |

Full descriptions: `skills_and_tools.md`.

---

## 4. Key Technical Decisions

### 4.1 JSON-constrained generation to prevent reasoning leakage

Qwen3 4B frequently produced visible chain-of-thought reasoning in its
output, even with `think=False` set and explicit prompt instructions not
to. This was most visible on longer, open-ended questions, where the
model would exhaust its token budget mid-reasoning and never reach a
clean final answer.

**Fix:** both the Knowledge Q&A (`rag/ask.py`) and Policy Analysis
(`skills/policy_analysis.py`) generation calls use Ollama's `format="json"`
parameter, which constrains the output grammar from the first generated
token. This proved more reliable than attempting to detect and strip
reasoning text after generation.

### 4.2 Regex-first calculator pipeline

Natural-language math questions are resolved in order of speed and
reliability:
1. Already a plain arithmetic expression (after light normalization)
2. A deterministic regex pattern (percentages, increases/decreases,
   value changes, monthly-to-annual conversion, etc.)
3. An LLM few-shot prompt that translates the question's meaning into an
   expression, without attempting to solve it

Every expression — regardless of which stage produced it — is validated
against an AST (Abstract Syntax Tree) before evaluation, allowing only
numeric constants and the operators `+ - * /`. The calculator never calls
Python's `eval()` on unvalidated input.

When a calculation is clearly intended but a required number is missing,
the system returns a message stating it cannot compute an answer, rather
than inventing a value.

### 4.3 Marker-based LLM output parsing

For the intent classifier and the calculator's LLM fallback, the model is
asked to wrap its answer in explicit `ANSWER_START` / `ANSWER_END`
markers. Output is only trusted if those markers are actually present;
otherwise the system falls back to deterministic keyword rules. This
avoids a failure mode encountered during development, where scanning a
model's raw (unmarked) output for a category name produced false matches,
since the model's own reasoning text tends to mention every candidate
category while "thinking out loud."

---

## 5. Known Limitations

- **Local LLM latency** — response time depends on the host machine.
  Routes that combine retrieval with LLM reasoning are noticeably slower
  than routes that resolve without any LLM call.
- **Hardware constraint** — developed and tested on an 8 GB RAM MacBook
  Air (M2); Qwen3 4B was chosen as a practical fit for this hardware.
- **Calculator extraction** — uncommon or ambiguous phrasing relies on
  the LLM fallback rather than a hard-coded pattern, and is not
  guaranteed to be correct on truly novel phrasing.
- **Single-turn interaction** — no conversational memory across turns.
- **Single-intent routing** — a question containing two independent
  requests (e.g. a knowledge question and a calculation in the same
  sentence) is routed to one primary intent rather than answering both.

Full details: `README.md`.

---

## 6. Testing

Test scenarios covering all eight demonstrated capabilities — knowledge
retrieval, skill selection, tool invocation, combined retrieval and tool
usage, source grounding, multi-step execution, unsupported-question
handling, and natural-language calculation — are documented in
`test_scenarios.md`, including two deliberate "trap" questions designed
to confirm the system declines to answer rather than fabricating a
response when required information is unavailable.

---

## 7. How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull qwen3:4b
ollama pull nomic-embed-text

python rag/build_index.py
streamlit run app.py
```

Requires Ollama installed and running locally.