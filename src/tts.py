from openai import OpenAI
import tempfile
import os

from src.models import get_tts_model


def text_to_speech(text: str, openai_api_key: str, voice: str = "nova") -> str:
    """Generate TTS audio and return path to temporary MP3 file."""
    client = OpenAI(api_key=openai_api_key)

    # Limit text length to avoid large API calls
    text = text[:4000]

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
