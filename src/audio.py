"""Assemblage des segments de synthèse vocale en une piste unique et lisible.

Pourquoi ne pas simplement coller des MP3 bout à bout : un MP3 porte en tête un
en-tête qui déclare sa durée. Concaténés, le fichier obtenu annonce la durée du
PREMIER segment seulement, et le lecteur du navigateur s'arrête là — les octets
suivants sont pourtant bien présents. L'audio n'est pas tronqué, il est mal
annoncé.

On demande donc du PCM brut, un flux sans en-tête que l'on peut concaténer sans
ambiguïté, puis on écrit un seul WAV avec un en-tête unique et exact. Si ffmpeg
est disponible sur la machine, on compresse ensuite en MP3 — le WAV pèse une
trentaine de mégaoctets pour dix minutes.
"""
import io
import shutil
import subprocess
import wave

# Format renvoyé par l'API TTS d'OpenAI en response_format="pcm".
SAMPLE_RATE = 24_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16 bits


def pcm_to_wav(
    pcm: bytes,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    sample_width: int = SAMPLE_WIDTH,
) -> bytes:
    """Enveloppe un flux PCM dans un WAV doté d'un en-tête unique et correct."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def wav_duration(wav_bytes: bytes) -> float:
    """Durée réelle du WAV, en secondes — lue dans le fichier, non estimée."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def pcm_duration(pcm: bytes) -> float:
    return len(pcm) / float(SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def wav_to_mp3(wav_bytes: bytes) -> bytes | None:
    """Compresse en MP3 via ffmpeg. Retourne None si ffmpeg est absent ou échoue."""
    if not ffmpeg_available():
        return None
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-i", "pipe:0", "-codec:a", "libmp3lame", "-q:a", "5", "-f", "mp3", "pipe:1"],
            input=wav_bytes,
            capture_output=True,
            timeout=300,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout if result.returncode == 0 and result.stdout else None


def format_duration(seconds: float | None) -> str:
    if not seconds:
        return "durée inconnue"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes} min {secs:02d} s" if minutes else f"{secs} s"


def mime_for(path: str) -> str:
    return "audio/wav" if str(path).lower().endswith(".wav") else "audio/mpeg"
