from openai import OpenAI
import re

from src.models import get_tts_model
from src.library import AUDIO_DIR, audio_key, find_audio, register_audio

# L'API TTS plafonne l'entrée : on découpe les textes longs plutôt que de les tronquer.
MAX_CHARS = 3800


def strip_markdown(text: str) -> str:
    """Retire la syntaxe Markdown pour que la voix ne lise pas les symboles."""
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)   # titres
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                  # gras
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*", r"\1", text)  # italique
    text = re.sub(r"`+([^`]+)`+", r"\1", text)                    # code
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)               # liens
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)  # puces
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)  # listes numérotées
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)      # citations
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)  # séparateurs
    text = re.sub(r"\|", " ", text)                               # tableaux
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_for_tts(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Découpe en segments lisibles, en coupant aux paragraphes puis aux phrases."""
    if len(text) <= max_chars:
        return [text]

    segments: list[str] = []
    current = ""

    for paragraph in text.split("\n\n"):
        # Un paragraphe seul trop long : on le redécoupe phrase par phrase.
        pieces = [paragraph]
        if len(paragraph) > max_chars:
            pieces = re.split(r"(?<=[.!?])\s+", paragraph)

        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    segments.append(current)
                # Une phrase unique dépassant la limite : coupe franche en dernier recours.
                while len(piece) > max_chars:
                    segments.append(piece[:max_chars])
                    piece = piece[max_chars:]
                current = piece

    if current:
        segments.append(current)
    return segments


def text_to_speech(
    text: str,
    openai_api_key: str,
    voice: str = "nova",
    theme: str = "",
    force: bool = False,
) -> dict:
    """Génère (ou récupère) l'audio d'une fiche et l'archive durablement.

    Retourne le descriptif de l'audio archivé : chemin, voix, modèle, date.
    """
    spoken = strip_markdown(text)
    model = get_tts_model()
    key = audio_key(spoken, voice, model)

    if not force:
        existing = find_audio(key)
        if existing:
            return existing

    client = OpenAI(api_key=openai_api_key)
    audio_bytes = b""
    for segment in split_for_tts(spoken):
        response = client.audio.speech.create(model=model, voice=voice, input=segment)
        audio_bytes += response.content

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIO_DIR / f"{key}.mp3"
    path.write_bytes(audio_bytes)

    return register_audio(key, theme or "Sans thème", voice, model, path)
