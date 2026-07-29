"""Découverte des modèles réellement accessibles par le projet OpenAI."""
import os
from openai import OpenAI


class ModelNotConfigured(RuntimeError):
    """Aucun modèle sélectionné : l'utilisateur doit en choisir un dans la sidebar."""


def list_available_models(api_key: str) -> tuple[list[str], list[str]]:
    """Retourne (modèles de chat, modèles d'embedding) accessibles par ce projet."""
    client = OpenAI(api_key=api_key)
    ids = sorted(m.id for m in client.models.list().data)

    embeddings = [m for m in ids if "embedding" in m]
    # On exclut les modèles non conversationnels (audio, image, moderation, etc.)
    excluded = ("embedding", "whisper", "tts", "dall-e", "moderation", "sora", "realtime")
    chat = [m for m in ids if not any(x in m for x in excluded)]

    return chat, embeddings


def list_tts_models(api_key: str) -> list[str]:
    """Modèles de synthèse vocale accessibles par ce projet."""
    client = OpenAI(api_key=api_key)
    ids = sorted(m.id for m in client.models.list().data)
    return [m for m in ids if "tts" in m or "audio-preview" in m]


def _selected(session_key: str, env_key: str) -> str:
    value = os.getenv(env_key, "")
    try:
        import streamlit as st
        value = st.session_state.get(session_key) or value
    except Exception:
        pass
    if not value:
        raise ModelNotConfigured(
            f"Aucun modèle sélectionné. Choisissez-en un dans la barre latérale "
            f"(ou définissez {env_key} dans votre .env)."
        )
    return value


def get_chat_model() -> str:
    return _selected("chat_model", "OPENAI_MODEL")


def get_embedding_model() -> str:
    return _selected("embed_model", "OPENAI_EMBEDDING_MODEL")


def get_tts_model() -> str:
    return _selected("tts_model", "OPENAI_TTS_MODEL")
