from openai import OpenAI
import tempfile
import os
import re

from src.models import get_tts_model


def strip_markdown(text: str) -> str:
    """Retire la syntaxe Markdown pour que la voix ne lise pas les symboles."""
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)   # titres
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                  # gras
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*", r"\1", text)  # italique
    text = re.sub(r"`+([^`]+)`+", r"\1", text)                    # code
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)               # liens
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)  # puces
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)      # citations
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)  # séparateurs
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_to_speech(text: str, openai_api_key: str, voice: str = "nova") -> str:
    """Generate TTS audio and return path to temporary MP3 file."""
    client = OpenAI(api_key=openai_api_key)

    # Limit text length to avoid large API calls
    text = strip_markdown(text)[:4000]

    response = client.audio.speech.create(
        model=get_tts_model(),
        voice=voice,
        input=text,
    )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    response.stream_to_file(tmp.name)
    return tmp.name


def cleanup_audio(path: str):
    try:
        os.unlink(path)
    except Exception:
        pass
