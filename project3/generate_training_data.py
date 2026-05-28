"""
Fine-Tuning Data Generation — generate_training_data.py

Generates synthetic Q&A pairs for fine-tuning a game recommendation model.
Uses Claude to produce high-quality training examples from our game database.

Output: training_data.jsonl  (instruction-following format for LoRA fine-tuning)

Run:
    python generate_training_data.py
"""

import json
import os
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Load game database
with open("data/games.json") as f:
    games = json.load(f)

SYSTEM_PROMPT = """You are a game recommendation expert. Generate realistic recruiter-style Q&A pairs for fine-tuning a game recommendation AI agent. Each pair should sound like a real conversation between a gamer and an expert friend."""

def generate_pairs_for_game(game: dict) -> list[dict]:
    """Generate 3-5 Q&A training pairs based on a single game profile."""

    game_summary = f"""
Title: {game['title']}
Genres: {', '.join(game['genres'])}
Tags: {', '.join(game['tags'])}
Difficulty: {game['difficulty']}
Playtime: {game['avg_playtime_hours']} hours
Multiplayer: {game['multiplayer']}
Description: {game['description'][:300]}
Best for: {game['best_for']}
Similar games: {', '.join(game['similar_games'][:3])}
"""

    prompt = f"""Based on this game profile, generate 4 diverse Q&A training pairs.

Game profile:
{game_summary}

Generate exactly 4 Q&A pairs in this JSON format:
[
  {{
    "question": "a natural question a gamer might ask",
    "answer": "an expert, conversational recommendation answer (2-4 sentences)"
  }},
  ...
]

Make questions diverse: some about the game directly, some comparing it, some about who it's for, some about difficulty or time commitment.
Return only valid JSON, nothing else."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Use Haiku for cost efficiency on data gen
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        pairs = json.loads(response.content[0].text)
        return pairs
    except json.JSONDecodeError:
        # Try to extract JSON from response
        text = response.content[0].text
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
        return []


def format_for_training(question: str, answer: str) -> dict:
    """Format a Q&A pair into instruction-following format for LoRA fine-tuning."""
    return {
        "instruction": "You are an expert game recommendation assistant. Recommend games based on the user's preferences, playing style, and interests. Be conversational, specific, and honest about both strengths and weaknesses.",
        "input": question,
        "output": answer,
        # Also include chat format for models that prefer it
        "messages": [
            {"role": "system", "content": "You are an expert game recommendation assistant. Recommend games based on the user's preferences, playing style, and interests. Be conversational, specific, and honest about both strengths and weaknesses."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    output_path = Path("training_data.jsonl")
    all_pairs = []

    print(f"Generating training data from {len(games)} games...")
    print("This will make API calls to Claude Haiku (low cost).\n")

    for i, game in enumerate(games):
        print(f"  [{i+1}/{len(games)}] Generating pairs for: {game['title']}")

        try:
            pairs = generate_pairs_for_game(game)
            for pair in pairs:
                formatted = format_for_training(pair["question"], pair["answer"])
                all_pairs.append(formatted)
            print(f"    Generated {len(pairs)} pairs")
        except Exception as e:
            print(f"    Error: {e}")

        # Rate limit courtesy pause
        time.sleep(0.5)

    # Add some cross-game comparison pairs manually
    comparison_pairs = [
        {
            "question": "What's the difference between Elden Ring and Dark Souls III? Which should I start with?",
            "answer": "If you're new to FromSoftware games, start with Elden Ring. It's the more accessible entry point — the open world means when you're stuck on a boss, you can just go explore somewhere else and come back stronger. Dark Souls III is more linear and punishing, but its boss fights are incredibly well-designed. Once you've finished Elden Ring, Dark Souls III feels like a tighter, more focused experience."
        },
        {
            "question": "I only have about 15 hours to play. What's the best game I can finish in that time?",
            "answer": "Outer Wilds is the perfect answer — it's around 15 hours and one of the most unique experiences in gaming. Go in completely blind, no guides. If you want something with more action, God of War (2018) is about 20-25 hours and absolutely worth every minute. Celeste is another great choice at 12 hours if you want a challenging platformer with a great story."
        },
        {
            "question": "My partner and I want to play something together on the couch. What do you recommend?",
            "answer": "Stardew Valley is genuinely perfect for couples — no stress, no fail state, you build a farm together at your own pace. It's become a classic 'couples game' for a reason. If you both want something with more action, Deep Rock Galactic is hilarious in co-op (you'll spend half the time accidentally killing each other). Valheim is great if you both enjoy survival games with a Viking theme."
        },
        {
            "question": "I've never played a souls-like before. Should I start with Elden Ring or something easier?",
            "answer": "Start with Elden Ring. I know that sounds counterintuitive, but the open world is actually more forgiving for beginners — when a boss destroys you, you can leave and explore somewhere else instead of being forced to face it. Hollow Knight is another great entry point if you want to understand the difficulty philosophy with 2D platformer mechanics before going full 3D. Avoid starting with Dark Souls 1 — the mechanics are dated and it'll put you off the genre."
        },
        {
            "question": "What's the best game for someone who loves story above everything else?",
            "answer": "Disco Elysium if you want the most literary, unique writing in any video game — it's unlike anything else and will stick with you for months. The Witcher 3 if you want 100+ hours of incredible story with excellent side quests. Outer Wilds if you want a mystery that unfolds through pure exploration. All three are masterclasses in storytelling, just in completely different ways."
        },
        {
            "question": "I loved Hades. What should I play next?",
            "answer": "Slay the Spire scratches a similar itch with deck-building roguelike runs — each attempt feels different and the synergy discovery is incredibly satisfying. Dead Cells is another excellent option if you want faster action. If you want more of the story-driven roguelike angle that Hades nailed, there isn't really another game that does it as well — but Transistor and Pyre (also by Supergiant) have that same beautiful art direction and strong narrative."
        },
    ]

    for pair in comparison_pairs:
        all_pairs.append(format_for_training(pair["question"], pair["answer"]))

    # Write JSONL
    with open(output_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nDone! Generated {len(all_pairs)} training pairs.")
    print(f"Saved to: {output_path}")
    print(f"\nFile size: {output_path.stat().st_size / 1024:.1f} KB")

    # Also save a small sample for inspection
    sample_path = Path("training_data_sample.json")
    with open(sample_path, "w") as f:
        json.dump(all_pairs[:5], f, indent=2)
    print(f"Sample (first 5): {sample_path}")


if __name__ == "__main__":
    main()
