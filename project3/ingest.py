"""
Data Engineering Module — ingest.py

Loads game profiles from JSON, converts them to rich searchable text documents,
embeds them, and stores them in a local ChromaDB vector database.

Run once before starting app.py:
    python ingest.py
"""

import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

DATA_PATH = Path("data/games.json")
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "games"


def game_to_document(g: dict) -> str:
    """Convert a structured game dict into a rich text document for embedding."""
    lines = [
        f"Game: {g['title']}",
        f"Developer: {g['developer']} | Publisher: {g['publisher']} | Year: {g['release_year']}",
        f"Genres: {', '.join(g['genres'])}",
        f"Platforms: {', '.join(g['platforms'])}",
        f"Tags: {', '.join(g['tags'])}",
        f"Difficulty: {g['difficulty']} | Avg playtime: {g['avg_playtime_hours']} hours | Multiplayer: {'Yes' if g['multiplayer'] else 'No'}",
        f"Metacritic: {g['metacritic_score']}/100",
        "",
        f"Description: {g['description']}",
        "",
        f"Why people love it: {g['why_people_love_it']}",
        f"Why people dislike it: {g['why_people_dislike_it']}",
        "",
        f"Similar games: {', '.join(g['similar_games'])}",
        f"Best for: {g['best_for']}",
    ]
    return "\n".join(lines)


def build_metadata(g: dict) -> dict:
    return {
        "title": g["title"],
        "genres": ", ".join(g["genres"]),
        "tags": ", ".join(g["tags"]),
        "difficulty": g["difficulty"],
        "avg_playtime_hours": str(g["avg_playtime_hours"]),
        "multiplayer": str(g["multiplayer"]),
        "metacritic_score": str(g["metacritic_score"]),
        "platforms": ", ".join(g["platforms"]),
        "release_year": str(g["release_year"]),
    }


def ingest():
    print(f"Loading games from {DATA_PATH} ...")
    with open(DATA_PATH, "r") as f:
        games = json.load(f)
    print(f"  Loaded {len(games)} games.")

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids, documents, metadatas = [], [], []
    for g in games:
        ids.append(g["id"])
        documents.append(game_to_document(g))
        metadatas.append(build_metadata(g))

    print("  Embedding and storing documents ...")
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"  Done. {collection.count()} games stored in '{CHROMA_PATH}/'.")
    return collection.count()


if __name__ == "__main__":
    ingest()
