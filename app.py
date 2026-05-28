"""
Recruiter Q&A AI Agent — app.py

End-to-end RAG agent that lets recruiters ask natural language questions
about students and receive grounded, accurate answers.

Architecture:
  Query → ChromaDB retrieval (sentence-transformers embeddings)
        → Retrieved student profiles as context
        → Claude (Anthropic) answer grounded in context
        → Gradio chat interface

Run:
    python app.py
"""

import os
from pathlib import Path

import anthropic
import chromadb
import gradio as gr
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "students"
TOP_K = 3  # number of students to retrieve per query
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a professional talent intelligence assistant helping recruiters evaluate student candidates.

Your job:
- Answer recruiter questions accurately using ONLY the student profiles provided in the context below.
- Be specific: cite the student's name, school, skills, and relevant experience when relevant.
- If a question asks you to compare students, be fair and factual.
- If the answer is not in the provided profiles, say "I don't have enough information in the current candidate pool to answer that."

Safety and alignment rules you MUST follow:
1. NEVER invent or assume credentials, skills, or experience that are not explicitly stated in the profiles.
2. NEVER make recommendations based on race, gender, age, name origin, or any protected characteristic.
3. NEVER speculate about a candidate's personal life, immigration status, or anything not in their profile.
4. If a question asks you to rank candidates in a discriminatory way, politely decline and redirect to skills/experience.
5. Keep answers professional and appropriate for a hiring context.

Respond in clear, professional prose. Use bullet points for comparisons."""


# ── Vector DB setup ──────────────────────────────────────────────────────────
def load_collection():
    if not Path(CHROMA_PATH).exists():
        raise RuntimeError(
            "Vector database not found. Please run `python ingest.py` first."
        )
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


collection = load_collection()
anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ── RAG pipeline ─────────────────────────────────────────────────────────────
def retrieve(query: str, n_results: int = TOP_K) -> list[dict]:
    """Retrieve the top-k most relevant student profiles for the query."""
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({"document": doc, "metadata": meta, "score": 1 - dist})
    return retrieved


def build_context(retrieved: list[dict]) -> str:
    """Format retrieved profiles into a context block for the LLM."""
    parts = []
    for i, r in enumerate(retrieved, 1):
        parts.append(f"--- Candidate {i} (relevance: {r['score']:.2f}) ---\n{r['document']}")
    return "\n\n".join(parts)


def answer_query(query: str, history: list) -> str:
    """Full RAG pipeline: retrieve → build context → call Claude → return answer."""
    # Step 1: Retrieve relevant student profiles
    retrieved = retrieve(query)

    if not retrieved:
        return "No student profiles found. Please run `python ingest.py` first."

    context = build_context(retrieved)

    # Step 2: Build messages for Claude
    messages = []

    # Include conversation history for multi-turn context
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    # Add current query with retrieved context
    messages.append({
        "role": "user",
        "content": (
            f"Recruiter question: {query}\n\n"
            f"Retrieved student profiles:\n\n{context}"
        ),
    })

    # Step 3: Call Claude
    response = anthropic_client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    return response.content[0].text


# ── Gradio UI ────────────────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "Which students have experience with PyTorch?",
    "Who is the best candidate for a data engineering role?",
    "Do any students have startup or internship experience at big tech companies?",
    "Which students are graduating in 2025 and available for full-time roles?",
    "Who has experience with machine learning and is open to relocation?",
    "Compare the top two candidates for a backend engineering role.",
    "Which student has the highest GPA?",
    "Are there any students with robotics or computer vision experience?",
]

with gr.Blocks(
    title="Recruiter Q&A Agent",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
    .header-text { text-align: center; margin-bottom: 10px; }
    .disclaimer { font-size: 0.85em; color: #666; margin-top: 8px; }
    """,
) as demo:
    gr.Markdown(
        """
        # Recruiter Q&A Agent
        ### AI-powered talent intelligence — ask any question about the student candidate pool
        """,
        elem_classes="header-text",
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=520,
                show_copy_button=True,
                bubble_full_width=False,
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question about candidates...",
                    label="",
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Ask", variant="primary", scale=1)
            clear_btn = gr.Button("Clear conversation", variant="secondary", size="sm")

            gr.Markdown(
                "**Note:** Answers are grounded in retrieved student profiles. "
                "The agent will not invent credentials or make discriminatory comparisons.",
                elem_classes="disclaimer",
            )

        with gr.Column(scale=1):
            gr.Markdown("### Example Questions")
            for q in EXAMPLE_QUESTIONS:
                btn = gr.Button(q, size="sm", variant="secondary")
                btn.click(lambda x=q: x, outputs=msg_input)

    # State
    history_state = gr.State([])

    def user_submit(user_message, history):
        if not user_message.strip():
            return history, history, ""
        answer = answer_query(user_message, history)
        history = history + [[user_message, answer]]
        return history, history, ""

    send_btn.click(
        user_submit,
        inputs=[msg_input, history_state],
        outputs=[chatbot, history_state, msg_input],
    )
    msg_input.submit(
        user_submit,
        inputs=[msg_input, history_state],
        outputs=[chatbot, history_state, msg_input],
    )

    def clear():
        return [], [], ""

    clear_btn.click(clear, outputs=[chatbot, history_state, msg_input])

if __name__ == "__main__":
    print("Starting Recruiter Q&A Agent...")
    print(f"  Model: {MODEL}")
    print(f"  Students in database: {collection.count()}")
    demo.launch(share=False)
