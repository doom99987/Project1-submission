"""
Game Recommendation Agent — app.py

A conversational AI agent that recommends games based on what you love,
how you play, and what you're in the mood for.

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

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "games"
TOP_K = 4
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an expert game recommendation agent — think of yourself as that one friend who has played everything and gives genuinely great recommendations instead of just naming whatever is popular.

Your personality:
- Conversational, enthusiastic, and honest. You love games.
- You give SPECIFIC, OPINIONATED recommendations backed by real reasons.
- You ask follow-up questions when needed to understand the player better.
- You're honest about a game's downsides — no recommendation is perfect for everyone.

Your job:
- Recommend games from the provided game database using ONLY information from the retrieved game profiles.
- Match games to the player's: favorite games, preferred genres, difficulty tolerance, playtime availability, whether they want co-op/single player, and current mood.
- When recommending, explain WHY this specific person will like it — connect it to what they told you.
- If they mention a game not in your database, you can use it to infer their taste and find the closest match from the database.

Rules:
1. Base all factual claims (Metacritic score, playtime, platforms, features) ONLY on the retrieved game profiles.
2. If you don't have a perfect match, say so and explain what's closest and why.
3. If someone asks about a specific game not in the database, tell them you don't have detailed info on that one but offer to find something similar.
4. Keep recommendations focused — 1 to 3 games is better than a list of 10.
5. If you need more information to give a good recommendation, ask a specific question.

Format: conversational prose, not bullet lists, unless doing a direct comparison. Be the friend who gives real talk about games."""


def load_collection():
    if not Path(CHROMA_PATH).exists():
        raise RuntimeError("Vector database not found. Please run `python ingest.py` first.")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


collection = load_collection()
anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def retrieve(query: str, n_results: int = TOP_K) -> list[dict]:
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
    parts = []
    for i, r in enumerate(retrieved, 1):
        parts.append(f"--- Game {i} (relevance: {r['score']:.2f}) ---\n{r['document']}")
    return "\n\n".join(parts)


def answer_query(user_message: str, history: list) -> str:
    # Build a richer search query from conversation context
    search_query = user_message
    if history:
        last_user = history[-1]["content"] if history[-1]["role"] == "user" else ""
        search_query = f"{last_user} {user_message}".strip()

    retrieved = retrieve(search_query)
    context = build_context(retrieved)

    # Convert gradio history (list of dicts) to Anthropic message format
    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": (
            f"{user_message}\n\n"
            f"[Retrieved game profiles for context:]\n{context}"
        ),
    })

    response = anthropic_client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


EXAMPLE_PROMPTS = [
    "I loved Elden Ring, what should I play next?",
    "I want a cozy game to play with my partner",
    "Best game for someone with only 2 hours a week to play?",
    "I liked the story in The Last of Us, what's similar?",
    "I want something scary and atmospheric",
    "What's the best co-op game to play with friends?",
    "I've never played a souls-like — where do I start?",
    "I want a game I can sink 100+ hours into",
]

CSS = """
.game-header { text-align: center; padding: 10px 0; }
.tagline { text-align: center; color: #888; font-size: 0.95em; margin-bottom: 16px; }
"""

with gr.Blocks(title="Game Rec Agent") as demo:

    gr.Markdown("# Game Recommendation Agent", elem_classes="game-header")
    gr.Markdown(
        "Tell me what you've loved, what you're in the mood for, or how much time you have — I'll find your next game.",
        elem_classes="tagline",
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="",
                height=500,
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="What kind of game are you looking for?",
                    label="",
                    scale=5,
                    container=False,
                    autofocus=True,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Start over", size="sm", variant="secondary")

        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### Try asking...")
            for prompt in EXAMPLE_PROMPTS:
                btn = gr.Button(prompt, size="sm", variant="secondary")
                btn.click(lambda x=prompt: x, outputs=msg_input)

    history_state = gr.State([])

    def submit(user_message, history):
        if not user_message.strip():
            return history, history, ""
        answer = answer_query(user_message, history)
        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ]
        return history, history, ""

    send_btn.click(submit, [msg_input, history_state], [chatbot, history_state, msg_input])
    msg_input.submit(submit, [msg_input, history_state], [chatbot, history_state, msg_input])
    clear_btn.click(lambda: ([], [], ""), outputs=[chatbot, history_state, msg_input])

if __name__ == "__main__":
    print("Starting Game Recommendation Agent...")
    print(f"  Games in database: {collection.count()}")
    demo.launch(share=False, theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"))
