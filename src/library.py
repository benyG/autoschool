"""Persistance des fiches générées et des audios de synthèse vocale.

Une fiche et son audio sont indexés par empreinte de contenu : régénérer
exactement le même texte avec la même voix réutilise le MP3 déjà payé.
"""
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = "data/progress.db"
AUDIO_DIR = Path("data/audio")


def init_db():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            theme TEXT NOT NULL,
            model TEXT,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS verifications (
            content_hash TEXT PRIMARY KEY,
            theme TEXT NOT NULL,
            report TEXT NOT NULL,
            statut TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS audios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_hash TEXT NOT NULL UNIQUE,
            theme TEXT NOT NULL,
            voice TEXT NOT NULL,
            model TEXT,
            filename TEXT NOT NULL,
            size_bytes INTEGER,
            created_at TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()


def _hash(*parts: str) -> str:
    joined = "\x00".join(p or "" for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


# ── Fiches ────────────────────────────────────────────────────────────────────

def save_summary(theme: str, content: str, model: str) -> str:
    """Enregistre une fiche et retourne son empreinte."""
    init_db()
    content_hash = _hash(theme, content, model)
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT OR IGNORE INTO summaries (theme, model, content, content_hash, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (theme, model, content, content_hash, datetime.now().isoformat(timespec="seconds")),
    )
    con.commit()
    con.close()
    return content_hash


def get_latest_summary(theme: str) -> dict | None:
    """Dernière fiche enregistrée pour ce thème, ou None."""
    init_db()
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT content, content_hash, model, created_at FROM summaries "
        "WHERE theme = ? ORDER BY id DESC LIMIT 1",
        (theme,),
    ).fetchone()
    con.close()
    if not row:
        return None
    return {"content": row[0], "hash": row[1], "model": row[2], "created_at": row[3]}


def list_summaries() -> list[dict]:
    init_db()
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT theme, content_hash, model, created_at FROM summaries ORDER BY id DESC"
    ).fetchall()
    con.close()
    return [
        {"theme": r[0], "hash": r[1], "model": r[2], "created_at": r[3]} for r in rows
    ]


# ── Rapports de vérification ──────────────────────────────────────────────────

def save_verification(content_hash: str, theme: str, report: dict):
    """Un rapport vaut pour un contenu donné : régénérer la fiche l'invalide."""
    init_db()
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT OR REPLACE INTO verifications "
        "(content_hash, theme, report, statut, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            content_hash,
            theme,
            json.dumps(report, ensure_ascii=False),
            report.get("statut", "inconnu"),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    con.commit()
    con.close()


def get_verification(content_hash: str) -> dict | None:
    init_db()
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT report, created_at FROM verifications WHERE content_hash = ?",
        (content_hash,),
    ).fetchone()
    con.close()
    if not row:
        return None
    report = json.loads(row[0])
    report["verifie_le"] = row[1]
    return report


def verification_statuses() -> dict[str, str]:
    """Statut de vérification de la dernière fiche de chaque thème."""
    init_db()
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("""
        SELECT s.theme, v.statut FROM summaries s
        JOIN verifications v ON v.content_hash = s.content_hash
        WHERE s.id = (SELECT MAX(id) FROM summaries WHERE theme = s.theme)
    """).fetchall()
    con.close()
    return {r[0]: r[1] for r in rows}


# ── Audios ────────────────────────────────────────────────────────────────────

def audio_key(spoken_text: str, voice: str, model: str) -> str:
    return _hash(spoken_text, voice, model)


def find_audio(audio_hash: str) -> dict | None:
    """Retourne l'audio archivé si le fichier existe toujours sur disque."""
    init_db()
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT filename, theme, voice, model, size_bytes, created_at FROM audios "
        "WHERE audio_hash = ?",
        (audio_hash,),
    ).fetchone()
    con.close()
    if not row:
        return None
    path = AUDIO_DIR / row[0]
    if not path.exists():
        return None
    return {
        "path": str(path),
        "theme": row[1],
        "voice": row[2],
        "model": row[3],
        "size_bytes": row[4],
        "created_at": row[5],
        "hash": audio_hash,
    }


def register_audio(audio_hash: str, theme: str, voice: str, model: str, path: Path) -> dict:
    init_db()
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT OR REPLACE INTO audios "
        "(audio_hash, theme, voice, model, filename, size_bytes, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            audio_hash,
            theme,
            voice,
            model,
            path.name,
            path.stat().st_size,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    con.commit()
    con.close()
    return find_audio(audio_hash)


def list_audios(theme: str | None = None) -> list[dict]:
    """Audios archivés dont le fichier existe encore, du plus récent au plus ancien."""
    init_db()
    con = sqlite3.connect(DB_PATH)
    if theme:
        rows = con.execute(
            "SELECT audio_hash, filename, theme, voice, model, size_bytes, created_at "
            "FROM audios WHERE theme = ? ORDER BY id DESC",
            (theme,),
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT audio_hash, filename, theme, voice, model, size_bytes, created_at "
            "FROM audios ORDER BY id DESC"
        ).fetchall()
    con.close()

    result = []
    for r in rows:
        path = AUDIO_DIR / r[1]
        if path.exists():
            result.append({
                "hash": r[0],
                "path": str(path),
                "theme": r[2],
                "voice": r[3],
                "model": r[4],
                "size_bytes": r[5],
                "created_at": r[6],
            })
    return result


def delete_audio(audio_hash: str):
    init_db()
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT filename FROM audios WHERE audio_hash = ?", (audio_hash,)
    ).fetchone()
    if row:
        (AUDIO_DIR / row[0]).unlink(missing_ok=True)
        con.execute("DELETE FROM audios WHERE audio_hash = ?", (audio_hash,))
        con.commit()
    con.close()


def library_stats() -> dict:
    audios = list_audios()
    return {
        "audio_count": len(audios),
        "audio_bytes": sum(a["size_bytes"] or 0 for a in audios),
        "summary_count": len(list_summaries()),
    }
