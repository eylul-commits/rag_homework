# Final Report

## 1. Methodology

### 1.1 System architecture

The stack is **LangChain** (loaders, splitters, Chroma integration, LCEL) plus **Ollama** for embeddings and (by default) chat, with vectors stored in **Chroma** on disk.

- **Ingestion.** PDFs in `data/pdfs/` are read with `PyPDFLoader`, chunked with `RecursiveCharacterTextSplitter` (defaults: 1000 characters, 200 overlap), embedded with `OllamaEmbeddings` (default `nomic-embed-text`, env `OLLAMA_EMBED_MODEL`), and written to `chroma_db` in collection `rag_docs`.
- **Retrieval and generation.** The query uses the same embedding model, **similarity search** over Chroma, and returns the top `k` chunks (default 4, `top_k`). Their text is joined into one `context` string. An **LCEL** chain runs: `ChatPromptTemplate` → **ChatOllama** (env `OLLAMA_MODEL`, default `llama3`) or optionally **Gemini** → plain string output.
- **Prompts.** The system message frames the model as a teaching assistant, inserts `context`, and asks for answers **only** from that context, no invented facts, and concise replies; the user turn is the question alone (`src/prompts.py`).
- **Configuration.** Defaults and paths (chunking, `top_k`, collection name, provider, API keys) are merged in `get_config()` with overrides from `.env` and CLI where supported.

### 1.2 Evaluation dataset

- **Total questions:** 20 (8 easy, 8 medium, 4 hard)
- **Source:** Five PDFs ingested into the vector store, with four short-answer questions per paper: (1) *AgentStepper: Interactive Debugging of Software Development Agents* (Feb 2026)—q5–q8; (2) *ESAA: Event Sourcing for Autonomous Agents in LLM-Based Software Engineering* (Feb 2026)—q17–q20; (3) *Managing Uncertainty in LLM-based Multi-Agent System Operation* (Feb 2026)—q13–q16; (4) *SWE-World: Building Software Engineering Agents in Docker-Free Environments* (Feb 2026)—q9–q12; (5) *Rethinking the Value of Agent-Generated Tests for LLM-Based Software Engineering Agents* (Feb 2026)—q1–q4.

### 1.3 Evaluation method

- **LangSmith** with 4 LLM-as-judge evaluators: correctness, relevance, hallucination, conciseness.
- **Judge model:** llama3.1:8b via Ollama / gemini-2.0-flash
- **Scoring:** Each metric returns a float in [0, 1]; higher is better. Hallucination score of 1 means no unsupported claims.

---

## 2. Manual Testing (Part 1)

### 2.1 Setup


| Parameter            | Value                                                |
| -------------------- | ---------------------------------------------------- |
| LLM                  | `llama3.1:8b` (Ollama; `OLLAMA_MODEL` in `.env`)     |
| Embedding model      | `nomic-embed-text`                                   |
| Chunk size / overlap | 1000 / 200 (`python -m src.ingest --force` defaults) |
| Top-k                | 4 (pipeline default from config)                     |


### 2.2 Results


| #   | Question (short)                                                                                                                  | Difficulty | Verdict | Answer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| --- | --------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | What is AgentStepper?                                                                                                             | easy       | Partial | There is no specific information about "AgentStepper" in the provided context. However, it does mention an agent called "agent" which has three components: the core program that orchestrates interaction between LLM and environment (AgentStepper is not mentioned as a component of this agent). AgentStepper might be assumed to refer to this component, but without further information, I cannot confirm this.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 2   | What percentage of tasks does gpt-5.2 write tests for?                                                                            | easy       | Pass    | According to Table 1 in the context, gpt-5.2 writes tests in about 0.6% of tasks (3/500).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 3   | Explain how SWE-World maintains the agent–environment interaction loop without real execution.                                    | hard       | Fail    | I don't see any mention of "SWE-World" in the provided context. However, I can explain how AgentStepper maintains the agent-environment interaction loop as per section 3.5 API for Integrating into Agents. AgentStepper manages the execution state of the agent by supporting three distinct states: paused, stepping, and running. The "running" state enables the agent to execute continuously until the developer decides to pause or step through the execution.To maintain the agent-environment interaction loop without real execution, AgentStepper provides an API that allows developers to insert API calls at critical points in the agent program. This approach allows developers to simulate the interaction between the agent and its environment, enabling interactive debugging without actual execution. However, I must note that there is no explicit information on how SWE-World maintains the agent-environment interaction loop without real execution. |
| 4   | In 'Managing Uncertainty in LLM-based Multi-Agent System Operation', what are the four core mechanisms in the proposed framework? | medium     | Fail    | There is no mention of "Managing Uncertainty in LLM-based Multi-Agent System Operation" in the provided context. The actual paper being discussed is called "Event Sourcing for Autonomous Agents in LLM-Based Software Engineering", and it does not specifically mention a framework with four core mechanisms.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 5   | What does ESAA stand for?                                                                                                         | easy       | Fail    | It is not provided in the context.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |


**Pass rate:** 1 / 5

### 2.3 Observations

- For “What is AgentStepper?”, chunks from the correct paper were used but it emphasized the paper’s "three-component agent model" rather than a sentence that defines AgentStepper; the model speculated instead of stating it is an interactive debugger.
- Straightforward extraction from a table in the retrieved context worked well (gpt-5.2 test-writing rate).
- Cross-paper queries often failed: retrieval or chunk boundaries favored one document (ESAA), so questions about SWE-World or the uncertainty paper were answered with the wrong source or “not in context.”
- The model sometimes filled gaps by discussing an incorrect entity (AgentStepper instead of SWE-World), which is worse than abstaining when the target term is missing.

---

## 3. Baseline Results (Part 2)

LangSmith experiment: `rag_baseline-c0611bc8` (20 examples; same setup as §2.1).

### 3.1 Configuration


| Parameter       | Value              |
| --------------- | ------------------ |
| Chunk size      | 1000               |
| Chunk overlap   | 200                |
| Top-k           | 4                  |
| LLM             | `llama3.1:8b`      |
| Embedding model | `nomic-embed-text` |


### 3.2 Overall metrics


| Metric        | Mean score |
| ------------- | ---------- |
| Correctness   | 0.690      |
| Relevance     | 0.680      |
| Hallucination | 0.710      |
| Conciseness   | 0.620      |


### 3.3 Metrics by difficulty


| Difficulty | Correctness | Relevance | Hallucination | Conciseness |
| ---------- | ----------- | --------- | ------------- | ----------- |
| Easy       | 0.512       | 0.638     | 0.575         | 0.675       |
| Medium     | 0.850       | 0.713     | 0.850         | 0.637       |
| Hard       | 0.725       | 0.700     | 0.700         | 0.475       |


### 3.4 Observations

- **Overall balance.** Mean scores sit in a tight band (about 0.62–0.71): the pipeline isn’t failing any metric but isn’t outstanding in any either. **Conciseness** is the lowest overall (0.620), so answers tend to be somewhat too long or indirect.
- **“Easy” vs “medium” split.** **Medium** questions get the strongest correctness and hallucination scores (0.850 for both), while **easy** is the weakest on those same axes (0.512 / 0.575). There are only eight easy examples, so the results are not very reliable. Still, they match what I saw in manual checks: even "easy" cases can fail when the system picks the wrong paper or when the model hesitates instead of giving a clear answer.
- **Hard questions and length.** **Hard** has the **lowest conciseness** (0.475) while correctness and hallucination stay mid–high (0.725 / 0.700). This suggests that harder prompts lead to longer, less focused answers, even when the judge still sees them as mostly grounded or partly correct.
- **Consistency with 2.3.** The automated metrics match the manual findings. Mixing up documents or answering about the wrong item lowers correctness and hallucination scores. When the model uses the right source, scores improve, this likely explains why the medium group performs relatively better when retrieval and reasoning are cleaner.

### 3.5 Failure case analysis

### Failure 1 (`q12`, hard)

- **Question:** Explain how SWE-World maintains the agent–environment interaction loop without real execution.
- **Expected:** It preserves the loop by using a sandbox for deterministic operations and LLM-based models (SWT and SWR) to simulate execution feedback and evaluation.
- **RAG answer:** The model states it cannot find any information about “SWE-World” in the provided context and asks for clarification, producing no substantive answer.
- **Scores:** correctness=0 (no answer provided about SWE-World’s mechanism); relevance=0 (response does not address the question); hallucination=0 (the model’s claim is not backed by the context); conciseness=0 (LangSmith reference_example_id `ca4fdc5-134a-413e-907c-4ebe5117a582`). **All zeros** because the output is a **complete refusal**, it provides **no information** about SWE-World’s sandbox, SWT, or SWR components, and **fails to address** the question entirely.
- **Root cause:** **Complete retrieval failure.** The retriever fetched four chunks entirely about AgentStepper (a debugging tool for LLM-based agents) instead of SWE-World. With zero relevant context, the model correctly identified the term was absent but could not produce any answer, resulting in all-zero scores. This is a pure recall problem: the relevant SWE-World passages were never surfaced to the model.

#### Failure 2 (`q16`, easy)

- **Question:** What are the three types of ontological uncertainty?
- **Expected:** Aleatory uncertainty, architectural morphing, and interaction uncertainty.
- **RAG answer:** It explains aleatory uncertainty correctly, then conflates epistemological and ontological categories and wrongly concludes that only one ontological type is clearly defined, failing to list all three.
- **Scores:** correctness=0 (contradicts the reference by claiming only one type exists); relevance=0.8 (identifies aleatory uncertainty but misses the other two); hallucination=0 (makes unsupported claims about epistemological uncertainty being an ontological type); conciseness=0.6 (LangSmith reference_example_id `c6defc24-cbdf-4597-82c6-25314ee4f1f5`, example `q16`). The **zeros** are because the output **contradicts the reference** (says only one type is mentioned), **does not complete the asked list**, and contains **claims not supported by the retrieved context**. Relevance **0.8** reflects that it does correctly identify aleatory uncertainty. Conciseness **0.6** is compatible with a **moderately sized** answer that includes unnecessary detail about epistemological uncertainty.
- **Root cause:** **Fragmented context + reasoning slip.** The retriever surfaced the right section but only the passage about aleatory uncertainty was detailed enough for the model to latch onto. The model saw aleatory explained clearly but lost track of the other two types (architectural morphing and interaction uncertainty). It then fixated on that one example and incorrectly concluded the others aren't given, instead of listing all three or abstaining.

#### Failure 3 (`q14`, medium)

- **Question:** In ‘Managing Uncertainty in LLM-based Multi-Agent System Operation’, what are the four core mechanisms in the proposed framework?
- **Expected:** Representation, identification, evolution, and adaptation.
- **RAG answer:** The model states it cannot find any information about the paper and admits it does not have enough context to answer. It mentions AgentStepper’s structured representation in passing but never identifies the four mechanisms.
- **Scores:** correctness=0 (no answer about the four mechanisms); relevance=0 (does not address the question); hallucination=0 (makes claims not backed by the context); conciseness=0.8 (LangSmith reference_example_id `131fcb5e-a349-4996-b081-b721f0eb035b`, example `q14`). The **zeros** are because the output is a **refusal**, it provides **no information** about representation, identification, evolution, or adaptation, and **fails to address** the question entirely. Conciseness **0.8** reflects a short reply, brevity is its one strength.
- **Root cause:** **Complete retrieval failure.** The retriever returned four chunks from unrelated papers (ESAA, AgentStepper, and reference lists) instead of the uncertainty-management paper. With zero relevant context, the model correctly identified the mismatch but could not produce any answer. This is a pure recall problem: the relevant passages about the four-mechanism framework were never surfaced.

---

## 4. Experiment Results (Part 3)

### 4.1 Configurations tested


| #   | Chunk size | Overlap | Top-k | Experiment slug |
| --- | ---------- | ------- | ----- | --------------- |
| 0   | 500        | 50      | 3     | cs500_co50_k3   |
| 1   | 500        | 100     | 5     | cs500_co100_k5  |
| 2   | 800        | 100     | 4     | cs800_co100_k4  |
| 3   | 1000       | 100     | 3     | cs1000_co100_k3 |
| 4   | 1000       | 200     | 5     | cs1000_co200_k5 |
| 5   | 1000       | 200     | 7     | cs1000_co200_k7 |
| 6   | 1500       | 200     | 3     | cs1500_co200_k3 |
| 7   | 1500       | 300     | 5     | cs1500_co300_k5 |
| 8   | 2000       | 200     | 3     | cs2000_co200_k3 |
| 9   | 2000       | 400     | 5     | cs2000_co400_k5 |
| 10  | 2000       | 400     | 7     | cs2000_co400_k7 |


### 4.2 Results comparison


| Experiment      | Correctness | Relevance | Hallucination | Conciseness |
| --------------- | ----------- | --------- | ------------- | ----------- |
| cs500_co50_k3   | 0.700       | 0.685     | 0.680         | 0.645       |
| cs500_co100_k5  | 0.770       | 0.685     | 0.760         | 0.660       |
| cs800_co100_k4  | 0.805       | 0.645     | 0.710         | 0.695       |
| cs1000_co100_k3 | 0.595       | 0.640     | 0.655         | 0.665       |
| cs1000_co200_k5 | 0.725       | 0.780     | 0.750         | 0.670       |
| cs1000_co200_k7 | 0.805       | 0.795     | 0.690         | 0.680       |
| cs1500_co200_k3 | 0.540       | 0.635     | 0.705         | 0.745       |
| cs1500_co300_k5 | 0.630       | 0.635     | 0.520         | 0.615       |
| cs2000_co200_k3 | 0.570       | 0.600     | 0.540         | 0.625       |
| cs2000_co400_k5 | 0.645       | 0.670     | 0.650         | 0.510       |
| cs2000_co400_k7 | 0.625       | 0.640     | 0.565         | 0.640       |


### 4.3 Optimal configuration

- **Best config:** **cs1000_co200_k7**
- **Justification:** It **ties the highest mean correctness** in the grid (**0.805**, same as **cs800_co100_k4**) and achieves the **highest mean relevance** (**0.795**; next is **cs1000_co200_k5** at 0.780). Hallucination (**hallucination** criterion: higher = fewer unsupported claims) is **0.690**, and conciseness is **0.680**. It is stronger in all-metric balance than **cs800_co100_k4**, which matches on correctness but drops to **0.645** relevance. 
- **Tradeoffs observed:**
  - **Chunk size ~800 with overlap 100 and k=4** (**cs800_co100_k4**) maximizes correctness in this sweep but **sacrifices relevance** (0.645 vs 0.795 for **cs1000_co200_k7**), consistent with **tighter chunks helping factual overlap with the reference** while **sometimes missing broader question coverage**.
  - **Chunk size 1000, overlap 200:** raising **top_k** from **5** to **7** (**cs1000_co200_k5** → **cs1000_co200_k7**) **increases** mean correctness (0.725 → 0.805) and relevance (0.780 → 0.795) with a **modest drop** in the mean hallucination-grounding score (0.750 → 0.690), so **extra retrieved passages helped more than they added unsupported detail** on average for this corpus.
  - **Very large chunks (1500–2000)** associate with **weaker mean correctness** ( **cs1500_co200_k3** at 0.540, **cs2000_** rows at 0.57–0.645), suggesting **retrieved windows are often too coarse or noisy** for fine-grained QA even when conciseness or other scores fluctuate.
  - **Largest overlap + high k** (**cs2000_co400_k5/k7**) trends toward **lowest mean conciseness** in the grid (down to **0.510**), aligning with **longer, more redundant context** encouraging **wordy answers**.

---

## 5. Gemini Comparison (Bonus)

### 5.1 Setup

- **Config used:** cs1000_co200_k5 (grid index 4)
- **Gemini model:** gemini-2.5-flash
- **Ollama model (comparison):** llama3.1:8b

### 5.2 Results


| Metric                   | Ollama (llama3.1:8b) | Gemini (gemini-2.5-flash) |
| ------------------------ | -------------------- | ------------------------- |
| Correctness              | 0.740                | 0.775                     |
| Relevance                | 0.700                | 0.665                     |
| Hallucination            | 0.745                | 0.770                     |
| Conciseness              | 0.715                | 0.760                     |
| Avg latency per question | ~7.24 s              | ~7.26 s                   |
| Estimated cost           | free (local)         | around 0,022$             |


### 5.3 Analysis

- **Gemini leads on three of four metrics.** It scores higher on correctness (+0.035), hallucination-grounding (+0.025), and conciseness (+0.045). The gains are consistent but modest (roughly 3–6 percentage points) suggesting that the cloud model is slightly better.
- **Ollama leads on relevance** (0.700 vs 0.665). Llama3.1:8b answers tend to stay closer to the question's topic even when they are less precise, whereas Gemini occasionally drifts into tangential but factually correct detail. Relevance judge penalizes this.
- **Latency is effectively identical** (~7.2 s per question for both). Ollama runs entirely on the local GPU, so its latency is pure compute, Gemini's includes a network roundtrip to Google's API, yet the cloud model's faster inference offsets the added network time, producing a tie on this hardware setup.

### 5.4 Recommendation

For this teaching-assistant RAG use case, **Ollama (llama3.1:8b) is the recommended default**. While Gemini edges ahead on correctness and conciseness, the differences are small (< 5 pp), and Ollama's advantages, full data privacy (no queries leave the machine), zero API dependency, no rate-limit concerns, and identical latency, outweigh the marginal quality gap. For deployments where answer correctness is critical and privacy constraints are relaxed, Gemini 2.5 Flash is a strong alternative, especially given its free-tier availability and slightly better grounding scores.

---

## 6. Conclusion

The strongest retrieval setup in the sweep was **chunk size 1000, overlap 200, top‑k 7** (`cs1000_co200_k7`), which tied the best mean correctness (0.805) while achieving the highest relevance (0.795) on LangSmith’s four judges, with **llama3.1:8b** and **nomic-embed-text** embeddings. 

Key findings: cross-paper similarity search often surfaced the wrong PDF (hurting easy questions and cases like SWE‑World vs AgentStepper), very large chunks (1500–2000) degraded correctness, and LLM-as-judge scores sometimes diverged from ground truth (fluent generic answers or “faithful” reads of the wrong document). With more time, I would add **metadata aware retrieval** (grouping by document), **tighter prompts or post-processing for conciseness**, and **calibration of judges** (or human spot checks) on ambiguous rows.

**Note:** Screenshots of the LangSmith dashboard and related UI are stored under `[data/images/](../data/images/)` in this repository (experiment runs, evaluators, comparisons).