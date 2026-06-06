import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from services.metadata_service import extract_and_validate_metadata
from services.audio_service import transcribe_audio
from services.ai_service import analyze_claim

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PMFBY AI Intake Engine",
    description="Multimodal claim intake for Pradhan Mantri Fasal Bima Yojana",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Serve the single-page frontend
# ---------------------------------------------------------------------------
FRONTEND_PATH = Path(__file__).parent / "index.html"


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found")
    return FRONTEND_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Main claim-submission endpoint
# ---------------------------------------------------------------------------
@app.post("/submit-claim")
async def submit_claim(
    policy_number: str = Form(..., description="Farmer's PMFBY policy number"),
    image: UploadFile = File(..., description="Photo of the damaged field"),
    audio: UploadFile = File(..., description="Voice note describing the damage"),
):
    logger.info("Received claim submission for policy: %s", policy_number)

    # --- 1. Read raw bytes (we need them multiple times) -------------------
    image_bytes = await image.read()
    audio_bytes = await audio.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    # --- 2. EXIF metadata extraction + 72-hour window validation -----------
    logger.info("Extracting EXIF metadata from image...")
    metadata_result = extract_and_validate_metadata(image_bytes)

    # --- 3. Audio transcription via Google Cloud Speech-to-Text (Chirp v2) -
    logger.info("Transcribing audio note...")
    transcription_result = await transcribe_audio(audio_bytes, audio.content_type or "audio/webm")

    # --- 4. Multimodal AI analysis via Gemini 1.5 Flash -------------------
    logger.info("Sending multimodal payload to Gemini 1.5 Flash...")
    ai_result = await analyze_claim(
        image_bytes=image_bytes,
        transcribed_text=transcription_result["transcription"],
        metadata=metadata_result,
    )

    # --- 5. Assemble the final NCIP-aligned response payload ---------------
    # Mock policy fetch – replace with real PMFBY API call in production
    mock_policy = _fetch_mock_policy(policy_number)

    response_payload = {
        "status": "INTAKE_SUCCESS",
        "policy_details": mock_policy,
        "metadata_analysis": metadata_result,
        "transcription": transcription_result,
        "ncip_claim_payload": {
            **ai_result,
            "policy_number": policy_number,
            "farmer_name": mock_policy["farmer_name"],
            "village": mock_policy["village"],
            "district": mock_policy["district"],
            "state": mock_policy["state"],
            "incident_date": metadata_result.get("photo_datetime", "UNKNOWN"),
            "gps_coordinates": metadata_result.get("gps_coordinates"),
            "within_72_hours": metadata_result.get("within_72_hours", False),
            "submission_channel": "AI_INTAKE_ENGINE_v1",
        },
        "warnings": _build_warnings(metadata_result, ai_result),
    }

    logger.info("Claim intake complete. Peril detected: %s", ai_result.get("peril_type"))
    return response_payload


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "service": "PMFBY AI Intake Engine"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fetch_mock_policy(policy_number: str) -> dict:
    """
    Simulates fetching farmer policy details from the PMFBY/NCIP database.
    Replace this stub with a real authenticated API call in production.
    """
    return {
        "policy_number": policy_number,
        "farmer_name": "Rajesh Kumar",
        "father_name": "Suresh Kumar",
        "aadhaar_last4": "7823",
        "bank_account": "XXXX-XXXX-4521",
        "ifsc": "SBIN0001234",
        "insured_crop": "Paddy (Kharif)",
        "insured_area_hectares": 2.5,
        "village": "Rampur",
        "tehsil": "Barnala",
        "district": "Barnala",
        "state": "Punjab",
        "season": "Kharif 2026",
        "sum_insured": 125000,
    }


def _build_warnings(metadata: dict, ai: dict) -> list[str]:
    """Collects advisory warnings to surface in the UI."""
    warnings = []

    if not metadata.get("within_72_hours"):
        warnings.append(
            "⚠️  CRITICAL: The photo timestamp is outside the 72-hour reporting window. "
            "This claim may be ineligible. Escalate to manual review immediately."
        )
    if not metadata.get("exif_found"):
        warnings.append(
            "⚠️  No EXIF timestamp found in image. Date/time could not be auto-verified. "
            "Attach a supplementary timestamp declaration."
        )
    if not metadata.get("gps_coordinates"):
        warnings.append(
            "ℹ️  GPS data absent from photo. Manual geo-tagging may be required by the assessor."
        )
    if ai.get("requires_manual_review"):
        warnings.append(
            "⚠️  AI could not confidently classify the peril. Flagged for mandatory human assessor review."
        )
    if ai.get("estimated_damage_percentage", 0) >= 75:
        warnings.append(
            "🔴  Severe damage (≥75%) detected. Expedited joint survey may be triggered per PMFBY norms."
        )

    return warnings
