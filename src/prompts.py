from langchain_core.prompts import ChatPromptTemplate

RAG_SYSTEM = """You are a teaching assistant answering from the provided course materials.

Rules:
- Base your answer only on the context below. If the context does not contain enough information, say so clearly.
- Do not invent facts that are not supported by the context.
- Be concise and direct.

Context:
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM),
        ("human", "{question}"),
    ]
)

EVALUATOR_SYSTEM = """You evaluate answers from a retrieval-augmented system.
Respond with valid JSON only, no markdown: {{"score": <float between 0 and 1>, "reasoning": "<one short sentence>"}}"""

EVALUATOR_USER = """{criterion}

Question: {question}
Ground truth (reference): {ground_truth}
Model answer: {answer}
Retrieved context shown to the model:
{context}

Apply the criterion and return JSON."""

EVALUATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", EVALUATOR_SYSTEM),
        ("human", EVALUATOR_USER),
    ]
)

CRITERION_CORRECTNESS = (
    "Score how factually correct the model answer is versus the ground truth, "
    "given the question. 1 = fully correct or equivalent; 0 = wrong or contradicts the reference."
)

CRITERION_RELEVANCE = (
    "Score how well the model answer addresses the question. "
    "1 = on-topic and complete enough; 0 = off-topic or avoids the question."
)

CRITERION_HALLUCINATION = (
    "Score whether the model answer stays supported by the retrieved context. "
    "1 = no unsupported claims (or clearly says context is insufficient); "
    "0 = clear claims not backed by the context."
)

CRITERION_CONCISENESS = (
    "Score whether the answer is appropriately concise versus the ground truth in length and focus. "
    "1 = tight and not padded; 0 = rambling, repetitive, or far longer than needed."
)
