"""
ai_service.py
-------------
Key prompt-engineering design decisions:
  1. CLOSED-SET PERIL CLASSIFICATION  - model may only output one of the five
     PMFBY-defined localized perils. Ambiguous cases flagged for human review.
  2. CONFIDENCE GATING               - peril_confidence < 0.70 -> manual review.
  3. ZERO HALLUCINATION TOLERANCE    - model forbidden from inventing data.
  4. SCHEMA-LOCKED OUTPUT            - strict JSON schema + post-parse validator.
"""

import base64
import json
import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Auth : Vertex AI via service account (production / GCP) 
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
GCP_LOCATION   = os.getenv("GCP_LOCATION", "us-central1")

GEMINI_MODEL = "gemini-2.0-flash"

# REST endpoint templates
_AI_STUDIO_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
_VERTEX_URL = (
    "https://{location}-aiplatform.googleapis.com/v1/projects/{project}/"
    "locations/{location}/publishers/google/models/{model}:generateContent"
)

# ---------------------------------------------------------------------------
# PMFBY Localized Peril Taxonomy
# (Section 4.2, PMFBY Revised Operational Guidelines)
# ---------------------------------------------------------------------------
VALID_PERILS = frozenset(
    ["Hailstorm", "Landslide", "Inundation", "Cloud Burst", "Natural Fire"]
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a highly specialised AI assistant for the Indian Government's
Pradhan Mantri Fasal Bima Yojana (PMFBY) crop insurance scheme. Your sole task is to
analyse a photograph of a damaged agricultural field along with a farmer's voice note
(transcribed from a regional Indian language) and extract structured claim data.

## STRICT OUTPUT CONTRACT
You MUST respond with ONLY a single, valid JSON object. Do not include any markdown
fencing (```json), preamble, explanation, or trailing text. The response must be
parseable by json.loads() with no pre-processing.

## REQUIRED JSON SCHEMA
{
  "crop_type": "string - the primary crop visible or mentioned (e.g., Paddy, Wheat, Cotton, Sugarcane, Maize, Soybean, Mustard, Chickpea, Groundnut). Use 'Unknown' only if completely indeterminate.",
  "peril_type": "string - MUST be EXACTLY one of: Hailstorm | Landslide | Inundation | Cloud Burst | Natural Fire. Use 'REQUIRES_MANUAL_REVIEW' if and ONLY if the evidence is genuinely ambiguous.",
  "peril_confidence": "float between 0.0 and 1.0 - your confidence in the peril classification",
  "estimated_damage_percentage": "integer between 0 and 100 - estimated percentage of visible crop area showing damage. Be conservative.",
  "damage_description": "string - 1-2 sentences describing the visible damage in objective factual terms suitable for an insurance assessor",
  "crop_growth_stage": "string - one of: Seedling | Vegetative | Flowering | Grain Filling | Maturity | Harvested | Unknown",
  "incident_date": "string - the photo EXIF date if available, else 'NOT_AVAILABLE'",
  "requires_manual_review": "boolean - true if peril_type is REQUIRES_MANUAL_REVIEW or peril_confidence < 0.70 or image does not clearly show agricultural damage",
  "review_reason": "string or null - concise reason if requires_manual_review is true, otherwise null"
}

## CLASSIFICATION RULES

### Inundation
Visual: Standing water in field; flattened/submerged crop rows; waterlogged soil with
reflective surface; silting or mud deposits on plants.
Audio: flood, baarish (rain), paani (water), doob (submerged), barh, khet mein paani.

### Cloud Burst
Visual: Severe soil erosion channels; uprooted plants with exposed roots; mixed debris
on crop; patchy/localised damage distinct from generalised inundation.
Audio: Tez baarish, achaanak baarish (sudden rain), cloudburst.

### Hailstorm
Visual: Circular impact craters/puncture wounds on leaves; shredded leaf canopy; ice
pellet residue; stripped grain heads; uniform damage across exposed surfaces.
Audio: Ole/ola (hail), pathhar (stones), barf ke tukde, shilabrishti.

### Landslide
Visual: Large soil/rock mass movement; buried crop rows; displacement scar on slope;
debris field across farm.
Audio: Zameen khisak gayi, mitti aayi, pahad se, bhavsandal.

### Natural Fire
Visual: Charred/blackened crop stalks; scorched soil; ash residue; burnt stubble
radiating from ignition point; smoke damage discolouration.
Audio: Aag (fire), jal gaya (burned), khet mein aag lagi, dhuaan (smoke).

## ANTI-HALLUCINATION RULES
1. Pest damage (insect holes, fungal spots, yellowing) -> REQUIRES_MANUAL_REVIEW.
   Pest damage is NOT a PMFBY localized calamity.
2. Blurry, overexposed, or unclear images -> requires_manual_review: true.
3. Prioritize visual evidence over audio for crop_type identification.
4. estimated_damage_percentage must reflect visible area only. Do not extrapolate.
5. Non-agricultural scene (urban, indoor, sky) -> requires_manual_review: true,
   review_reason: "Image does not show an agricultural field."
"""

_USER_PROMPT_TEMPLATE = """Analyse the attached photograph of a damaged crop field and the
farmer's transcribed voice note below.

FARMER'S VOICE NOTE (transcribed from regional language):
\"\"\"{transcription}\"\"\"

PHOTO METADATA:
- Timestamp from EXIF: {exif_datetime}
- Hours since photo was taken: {hours_since}
- GPS coordinates: {gps_info}
- Within PMFBY 72-hour reporting window: {within_72h}

Based on both the image and the voice note, extract the structured claim data per the
JSON schema in your system instructions. Output ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def analyze_claim(
    image_bytes: bytes,
    transcribed_text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:

    if not GEMINI_API_KEY and GCP_PROJECT_ID == "your-gcp-project-id":
        logger.warning(
            "No GEMINI_API_KEY or GCP_PROJECT_ID configured. "
            "Returning mock result. Set GEMINI_API_KEY in your .env file."
        )
        return _mock_ai_result(metadata)

    try:
        return await _call_gemini_http(image_bytes, transcribed_text, metadata)
    except Exception as exc:
        logger.error("Gemini API call failed: %s", exc, exc_info=True)
        return {
            "crop_type": "Unknown",
            "peril_type": "REQUIRES_MANUAL_REVIEW",
            "peril_confidence": 0.0,
            "estimated_damage_percentage": 0,
            "damage_description": f"AI analysis failed: {exc}",
            "crop_growth_stage": "Unknown",
            "incident_date": metadata.get("photo_datetime", "NOT_AVAILABLE"),
            "requires_manual_review": True,
            "review_reason": f"AI service error: {str(exc)[:200]}",
        }


# ---------------------------------------------------------------------------
# Private: direct HTTP call to Gemini REST API
# ---------------------------------------------------------------------------

async def _call_gemini_http(
    image_bytes: bytes,
    transcribed_text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    
    # Build user prompt
    gps = metadata.get("gps_coordinates")
    gps_str = f"Lat {gps['lat']}, Lon {gps['lon']}" if gps else "Not available"

    user_text = _USER_PROMPT_TEMPLATE.format(
        transcription=transcribed_text or "[No audio transcription available]",
        exif_datetime=metadata.get("photo_datetime", "Not available"),
        hours_since=(
            f"{metadata['hours_since_incident']:.1f} hours"
            if metadata.get("hours_since_incident") is not None
            else "Not available"
        ),
        gps_info=gps_str,
        within_72h="YES" if metadata.get("within_72_hours") else "NO - OUTSIDE WINDOW",
    )

    # Base64-encode image
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Detect image MIME type from magic bytes
    mime_type = _detect_mime(image_bytes)

    # Build the Gemini request body (same structure for both endpoints)
    request_body = {
        "system_instruction": {
            "parts": [{"text": _SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_b64,
                        }
                    },
                    {"text": user_text},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.8,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json",
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH",
            }
        ],
    }

    # --- Choose auth method and endpoint ---
    if GEMINI_API_KEY:
        # Option A: Google AI Studio API key - simplest, free tier available
        url = f"{_AI_STUDIO_URL}?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        logger.info("Using Google AI Studio API key auth")
    else:
        # Option B: Vertex AI with OAuth2 token from service account
        token = await _get_vertex_token()
        url = _VERTEX_URL.format(
            location=GCP_LOCATION,
            project=GCP_PROJECT_ID,
            model=GEMINI_MODEL,
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        logger.info("Using Vertex AI OAuth2 auth for project %s", GCP_PROJECT_ID)

    logger.info(
        "Sending multimodal request to Gemini (image: %d bytes, prompt: %d chars)",
        len(image_bytes),
        len(user_text),
    )

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=request_body)

    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini API returned HTTP {response.status_code}: {response.text[:400]}"
        )

    resp_json = response.json()

    # Extract text from response
    try:
        raw_text = (
            resp_json["candidates"][0]["content"]["parts"][0]["text"].strip()
        )
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected Gemini response structure: {resp_json}") from exc

    logger.info("Raw Gemini response (first 500 chars): %s", raw_text[:500])
    return _parse_and_validate(raw_text, metadata)


async def _get_vertex_token() -> str:
    """
    Obtain a short-lived OAuth2 access token using Application Default Credentials.
    Requires: pip install google-auth and GOOGLE_APPLICATION_CREDENTIALS set.
    """
    try:
        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        return credentials.token
    except Exception as exc:
        raise RuntimeError(
            "Could not obtain Vertex AI credentials. "
            "Either set GEMINI_API_KEY for AI Studio, or set "
            "GOOGLE_APPLICATION_CREDENTIALS pointing to a service account key. "
            f"Original error: {exc}"
        ) from exc


def _detect_mime(image_bytes: bytes) -> str:
    """Detect image MIME type from magic bytes — avoids relying on file extension."""
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:4] in (b"RIFF", b"WEBP"):
        return "image/webp"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    # Default to JPEG — most phone photos are JPEG
    return "image/jpeg"


# ---------------------------------------------------------------------------
# Private: response parsing + schema validation
# ---------------------------------------------------------------------------

def _parse_and_validate(raw_text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Parse Gemini's JSON response and enforce the PMFBY peril taxonomy.
    """
    # Strip stray markdown fences if the model ignored the mime_type hint
    clean = re.sub(r"^```(?:json)?|```$", "", raw_text, flags=re.MULTILINE).strip()

    try:
        payload = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned non-JSON output: {exc}\nRaw: {raw_text[:300]}"
        )

    # Enforce closed-set peril
    peril = payload.get("peril_type", "REQUIRES_MANUAL_REVIEW")
    if peril not in VALID_PERILS and peril != "REQUIRES_MANUAL_REVIEW":
        logger.warning(
            "Gemini returned invalid peril '%s' -> overriding to REQUIRES_MANUAL_REVIEW",
            peril,
        )
        payload["peril_type"] = "REQUIRES_MANUAL_REVIEW"
        payload["requires_manual_review"] = True
        payload["review_reason"] = (
            f"AI returned unrecognised peril type '{peril}'. "
            "Manual assessor must classify the event."
        )

    # Clamp confidence to [0, 1]
    payload["peril_confidence"] = max(
        0.0, min(1.0, float(payload.get("peril_confidence", 0.5)))
    )

    # Clamp damage % to [0, 100]
    payload["estimated_damage_percentage"] = max(
        0, min(100, int(payload.get("estimated_damage_percentage", 0)))
    )

    # Confidence gating: low-confidence -> manual review
    if payload["peril_confidence"] < 0.70 and not payload.get("requires_manual_review"):
        payload["requires_manual_review"] = True
        payload["review_reason"] = (
            f"AI confidence too low ({payload['peril_confidence']:.0%}) for automatic "
            "classification. Human assessor review required."
        )

    # Inject EXIF date if model returned placeholder
    if payload.get("incident_date") in ("NOT_AVAILABLE", None, ""):
        payload["incident_date"] = metadata.get("photo_datetime", "NOT_AVAILABLE")

    return payload


# ---------------------------------------------------------------------------
# Private: local dev mock
# ---------------------------------------------------------------------------

def _mock_ai_result(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Returned when no API credentials are configured.
    Shows what a real successful response looks like so the UI can be developed.
    """
    return {
        "crop_type": "Paddy",
        "peril_type": "Inundation",
        "peril_confidence": 0.91,
        "estimated_damage_percentage": 65,
        "damage_description": (
            "Standing water visible across approximately 65% of the paddy field. "
            "Crop rows are partially submerged with visible silting on lower leaves, "
            "consistent with prolonged inundation following heavy rainfall."
        ),
        "crop_growth_stage": "Grain Filling",
        "incident_date": metadata.get("photo_datetime", "NOT_AVAILABLE"),
        "requires_manual_review": False,
        "review_reason": None,
        "_mock": True,
        "_mock_note": (
            "No API credentials found. Set GEMINI_API_KEY in your .env file "
            "to enable real AI analysis. Get a free key at: "
            "https://aistudio.google.com/app/apikey"
        ),
    }
