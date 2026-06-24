import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")

# Chirp v2 recognizer resource name pattern
CHIRP_RECOGNIZER = f"projects/{GCP_PROJECT_ID}/locations/us-central1/recognizers/_"

# Primary language + alternatives covering major Indian agricultural states
LANGUAGE_CODES = [
    "hi-IN",   # Hindi        (UP, MP, Bihar, Rajasthan, Haryana, Delhi NCR)
    "pa-IN",   # Punjabi      (Punjab)
    "mr-IN",   # Marathi      (Maharashtra)
    "te-IN",   # Telugu       (Andhra Pradesh, Telangana)
    "ta-IN",   # Tamil        (Tamil Nadu)
    "kn-IN",   # Kannada      (Karnataka)
    "ml-IN",   # Malayalam    (Kerala)
    "gu-IN",   # Gujarati     (Gujarat)
    "or-IN",   # Odia         (Odisha)
    "bn-IN",   # Bengali      (West Bengal, Assam)
    "en-IN",   # English      (code-mixed usage across all states)
]


async def transcribe_audio(audio_bytes: bytes, content_type: str) -> dict[str, Any]:

    try:
        return await _call_chirp_v2(audio_bytes, content_type)
    except ImportError:
        logger.warning(
            "google-cloud-speech not installed or GCP not configured. "
            "Returning mock transcription for local development."
        )
        return _mock_transcription()
    except Exception as exc:
        logger.error("Speech-to-Text API call failed: %s", exc)
        # Return a graceful fallback rather than crashing the pipeline
        return {
            "transcription": "[TRANSCRIPTION_FAILED] " + str(exc),
            "confidence": 0.0,
            "detected_language": "unknown",
            "is_fallback": True,
        }


async def _call_chirp_v2(audio_bytes: bytes, content_type: str) -> dict[str, Any]:

    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech

    import google.auth
    from google.auth.transport.requests import Request as AuthRequest
    credentials, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(AuthRequest())
    client = SpeechClient(credentials=credentials)

    # Determine audio encoding from MIME type
    encoding = _mime_to_encoding(content_type)

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=LANGUAGE_CODES,
        model="chirp",
        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True,
            enable_spoken_punctuation=False,
            # Profanity filter off - agricultural dialect may trigger false positives
            profanity_filter=False,
        ),
    )

    request = cloud_speech.RecognizeRequest(
        recognizer=CHIRP_RECOGNIZER,
        config=config,
        content=audio_bytes,
    )

    logger.info("Sending %d bytes of audio to Chirp v2...", len(audio_bytes))
    response = client.recognize(request=request)

    if not response.results:
        return {
            "transcription": "",
            "confidence": 0.0,
            "detected_language": "unknown",
            "is_fallback": False,
        }

    best = response.results[0].alternatives[0]
    detected_lang = (
        response.results[0].language_code
        if response.results[0].language_code
        else "hi-IN"
    )

    logger.info(
        "Transcription success. Detected language: %s. Confidence: %.2f",
        detected_lang,
        best.confidence,
    )

    return {
        "transcription": best.transcript,
        "confidence": round(best.confidence, 3),
        "detected_language": detected_lang,
        "is_fallback": False,
    }


def _mock_transcription() -> dict[str, Any]:

    return {
        "transcription": (
            "Kal raat bahut tez baarish hui aur mere do hectare ke dhan ke khet mein "
            "paani bhar gaya. Poori fasal doob gayi hai. Nuksan bahut zyada hua hai."
            # Translation: "Last night there was very heavy rain and my two hectares of "
            # "paddy field got flooded. The entire crop is submerged. The damage is very severe."
        ),
        "confidence": 0.95,
        "detected_language": "hi-IN",
        "is_fallback": True,
    }


def _mime_to_encoding(mime: str) -> str:
    """Map browser MIME types to GCP encoding constants (informational only; AutoDetect used)."""
    mapping = {
        "audio/webm": "WEBM_OPUS",
        "audio/ogg": "OGG_OPUS",
        "audio/wav": "LINEAR16",
        "audio/wave": "LINEAR16",
        "audio/flac": "FLAC",
        "audio/mp3": "MP3",
        "audio/mpeg": "MP3",
        "audio/mp4": "MP4",
    }
    for key, val in mapping.items():
        if key in mime.lower():
            return val
    return "ENCODING_UNSPECIFIED"
