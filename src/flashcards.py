import sqlite3
import json
from datetime import date, timedelta
from openai import OpenAI
from src.vector_store import search

DB_PATH = "data/progress.db"


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS flashcards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            theme TEXT,
            interval INTEGER DEFAULT 1,
            ease_factor REAL DEFAULT 2.5,
            next_review TEXT DEFAULT (date('now')),
            repetitions INTEGER DEFAULT 0
        )
    """)
    con.commit()
    con.close()


def get_due_cards(limit: int = 20) -> list[dict]:
    init_db()
    today = date.today().isoformat()
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, question, answer, theme FROM flashcards WHERE next_review <= ? LIMIT ?",
        (today, limit),
    ).fetchall()
    con.close()
    return [{"id": r[0], "question": r[1], "answer": r[2], "theme": r[3]} for r in rows]


def review_card(card_id: int, quality: int):
    """SM-2 algorithm. quality: 0=fail, 1=hard, 2=ok, 3=easy"""
    init_db()
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT interval, ease_factor, repetitions FROM flashcards WHERE id = ?", (card_id,)
    ).fetchone()
    if not row:
        con.close()
        return

    interval, ef, reps = row

    if quality < 1:
        interval = 1
        reps = 0
    else:
        if reps == 0:
            interval = 1
        elif reps == 1:
            interval = 6
        else:
            interval = round(interval * ef)
        reps += 1
        ef = max(1.3, ef + 0.1 - (3 - quality) * (0.08 + (3 - quality) * 0.02))

    next_review = (date.today() + timedelta(days=interval)).isoformat()
    con.execute(
        "UPDATE flashcards SET interval=?, ease_factor=?, repetitions=?, next_review=? WHERE id=?",
        (interval, ef, reps, next_review, card_id),
    )
    con.commit()
    con.close()


def generate_flashcards(theme: str, openai_api_key: str, count: int = 8) -> list[dict]:
    client = OpenAI(api_key=openai_api_key)
    context_chunks = search(theme, openai_api_key, n_results=5)
    context = "\n\n".join(context_chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un expert en code de la route au Québec. "
                    "Tu génères des flashcards d'apprentissage précises basées sur le manuel officiel SAAQ. "
                    "Réponds UNIQUEMENT en JSON valide."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Génère {count} flashcards sur le thème : {theme}\n\n"
                    f"Extraits du manuel :\n{context}\n\n"
                    'Réponds avec ce JSON exact :\n{"cards": [{"question": "...", "answer": "..."}]}'
                ),
            },
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)
    return data.get("cards", [])


def save_flashcards(cards: list[dict], theme: str):
    init_db()
    con = sqlite3.connect(DB_PATH)
    for card in cards:
        existing = con.execute(
            "SELECT id FROM flashcards WHERE question = ?", (card["question"],)
        ).fetchone()
        if not existing:
            con.execute(
                "INSERT INTO flashcards (question, answer, theme) VALUES (?, ?, ?)",
                (card["question"], card["answer"], theme),
            )
    con.commit()
    con.close()


def get_stats() -> dict:
    init_db()
    today = date.today().isoformat()
    con = sqlite3.connect(DB_PATH)
    total = con.execute("SELECT COUNT(*) FROM flashcards").fetchone()[0]
    due = con.execute(
        "SELECT COUNT(*) FROM flashcards WHERE next_review <= ?", (today,)
    ).fetchone()[0]
    mastered = con.execute(
        "SELECT COUNT(*) FROM flashcards WHERE repetitions >= 3"
    ).fetchone()[0]
    con.close()
    return {"total": total, "due": due, "mastered": mastered}
