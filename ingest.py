"""
Data Engineering Module — ingest.py

Loads student profiles from JSON, converts them to rich text documents,
chunks and embeds them, then stores them in a local ChromaDB vector database.

Run this once before starting app.py:
    python ingest.py
"""

import json
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

DATA_PATH = Path("data/student_profiles.json")
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "students"


def profile_to_document(p: dict) -> str:
    """Convert a structured student profile dict into a rich text document for embedding."""
    lines = [
        f"Student: {p['name']}",
        f"Degree: {p['degree']} at {p['university']} (Class of {p['graduation_year']})",
        f"GPA: {p['gpa']}",
        f"Skills: {', '.join(p['skills'])}",
        f"Looking for: {p['seeking']}",
        f"Location: {p['location']}",
        f"Summary: {p['summary']}",
        "",
        "--- Work Experience ---",
    ]

    for exp in p.get("experience", []):
        lines.append(
            f"  • {exp['role']} at {exp['company']} ({exp['duration']}): {exp['description']}"
        )

    lines.append("")
    lines.append("--- Projects ---")

    for proj in p.get("projects", []):
        lines.append(f"  • {proj['name']}: {proj['description']}")

    lines.append(f"\nGitHub: {p.get('github', 'N/A')}")
    lines.append(f"LinkedIn: {p.get('linkedin', 'N/A')}")

    return "\n".join(lines)


def build_metadata(p: dict) -> dict:
    """Extract filterable metadata fields from a profile."""
    return {
        "name": p["name"],
        "university": p["university"],
        "degree": p["degree"],
        "graduation_year": str(p["graduation_year"]),
        "gpa": str(p["gpa"]),
        "skills": ", ".join(p["skills"]),
        "location": p["location"],
    }


def ingest():
    print(f"Loading profiles from {DATA_PATH} ...")
    with open(DATA_PATH, "r") as f:
        profiles = json.load(f)

    print(f"  Loaded {len(profiles)} student profiles.")

    # Use a local sentence-transformers model for embeddings — no API cost
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Drop and recreate collection for a clean ingest
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    documents = []
    metadatas = []

    for p in profiles:
        doc = profile_to_document(p)
        meta = build_metadata(p)
        ids.append(p["id"])
        documents.append(doc)
        metadatas.append(meta)

    print("  Embedding and storing documents in ChromaDB ...")
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"  Done. {collection.count()} documents stored in '{CHROMA_PATH}/'.")
    return collection.count()


if __name__ == "__main__":
    ingest()
