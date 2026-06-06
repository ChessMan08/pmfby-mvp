"""
tests/test_services.py
----------------------
Unit tests for all three PMFBY service modules.
Runs entirely offline - no GCP credentials required.
All external API calls are mocked via unittest.mock.patch.

Run with:
    pytest tests/test_services.py -v
"""

import io
import json
import struct
import zlib
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

# -----------------------------------------------------------------------------
# Helpers: build synthetic test images
# -----------------------------------------------------------------------------

def _make_jpeg_with_exif(dt_str: str, lat: float = 30.12, lon: float = 75.88) -> bytes:

    img = Image.new("RGB", (100, 80), color=(80, 120, 40))
    buf = io.BytesIO()

    try:
        import piexif  # optional dep for test helpers
        exif_dict = {
            "Exif": {piexif.ExifIFD.DateTimeOriginal: dt_str.encode()},
            "GPS": {
                piexif.GPSIFD.GPSLatitudeRef: b"N",
                piexif.GPSIFD.GPSLatitude:    _dd_to_dms_rational(lat),
                piexif.GPSIFD.GPSLongitudeRef: b"E",
                piexif.GPSIFD.GPSLongitude:   _dd_to_dms_rational(lon),
            },
        }
        exif_bytes = piexif.dump(exif_dict)
        img.save(buf, format="JPEG", exif=exif_bytes)
    except ImportError:
        img.save(buf, format="JPEG")

    return buf.getvalue()


def _make_plain_jpeg() -> bytes:
    img = Image.new("RGB", (64, 48), color=(34, 85, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _dd_to_dms_rational(dd: float):
    """Convert decimal degrees to DMS tuple of IFDRationals for piexif."""
    d = int(dd)
    m = int((dd - d) * 60)
    s = ((dd - d) * 60 - m) * 60
    return [(d, 1), (m, 1), (int(s * 1000), 1000)]


# -----------------------------------------------------------------------------
# metadata_service tests
# -----------------------------------------------------------------------------

class TestMetadataService:
    """Tests for services/metadata_service.py"""

    def test_no_exif_returns_graceful_defaults(self):
        from services.metadata_service import extract_and_validate_metadata
        result = extract_and_validate_metadata(_make_plain_jpeg())
        assert result["exif_found"] is False
        assert result["photo_datetime"] is None
        assert result["within_72_hours"] is False
        assert result["gps_coordinates"] is None

    def test_within_72_hour_window(self):
        from services.metadata_service import extract_and_validate_metadata
        # Photo taken 10 hours ago in IST
        now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
        ten_hours_ago = now_ist - timedelta(hours=10)
        dt_str = ten_hours_ago.strftime("%Y:%m:%d %H:%M:%S")
        jpeg_bytes = _make_jpeg_with_exif(dt_str)
        result = extract_and_validate_metadata(jpeg_bytes)
        if result["exif_found"]:  # piexif available
            assert result["within_72_hours"] is True
            assert result["hours_since_incident"] < 72

    def test_outside_72_hour_window(self):
        from services.metadata_service import extract_and_validate_metadata
        now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
        old_dt = now_ist - timedelta(hours=100)
        dt_str = old_dt.strftime("%Y:%m:%d %H:%M:%S")
        jpeg_bytes = _make_jpeg_with_exif(dt_str)
        result = extract_and_validate_metadata(jpeg_bytes)
        if result["exif_found"]:
            assert result["within_72_hours"] is False
            assert result["hours_since_incident"] > 72

    def test_invalid_image_bytes_returns_defaults(self):
        from services.metadata_service import extract_and_validate_metadata
        result = extract_and_validate_metadata(b"not-an-image")
        assert result["exif_found"] is False

    def test_empty_bytes_returns_defaults(self):
        from services.metadata_service import extract_and_validate_metadata
        result = extract_and_validate_metadata(b"")
        assert result["exif_found"] is False


# -----------------------------------------------------------------------------
# audio_service tests
# -----------------------------------------------------------------------------

class TestAudioService:
    """Tests for services/audio_service.py"""

    @pytest.mark.asyncio
    async def test_fallback_when_sdk_not_available(self):
        """If google-cloud-speech is not importable, should return mock transcription."""
        from services.audio_service import transcribe_audio
        with patch("services.audio_service._call_chirp_v2", side_effect=ImportError("not installed")):
            result = await transcribe_audio(b"fake-audio", "audio/webm")
        assert result["is_fallback"] is True
        assert len(result["transcription"]) > 0
        assert isinstance(result["confidence"], float)

    @pytest.mark.asyncio
    async def test_successful_transcription(self):
        """Mock a successful Chirp v2 API response."""
        mock_alternative = MagicMock()
        mock_alternative.transcript = "Mere khet mein paani bhar gaya hai"
        mock_alternative.confidence = 0.93

        mock_result = MagicMock()
        mock_result.alternatives = [mock_alternative]
        mock_result.language_code = "hi-IN"

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        with patch("services.audio_service._call_chirp_v2", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {
                "transcription": "Mere khet mein paani bhar gaya hai",
                "confidence": 0.93,
                "detected_language": "hi-IN",
                "is_fallback": False,
            }
            from services.audio_service import transcribe_audio
            result = await transcribe_audio(b"real-audio-bytes", "audio/webm")

        assert result["transcription"] == "Mere khet mein paani bhar gaya hai"
        assert result["confidence"] == 0.93
        assert result["detected_language"] == "hi-IN"
        assert result["is_fallback"] is False

    @pytest.mark.asyncio
    async def test_api_error_returns_graceful_fallback(self):
        """API errors should not crash the pipeline — return error string."""
        with patch("services.audio_service._call_chirp_v2", new_callable=AsyncMock,
                   side_effect=Exception("network timeout")):
            from services.audio_service import transcribe_audio
            result = await transcribe_audio(b"audio", "audio/webm")
        assert result["is_fallback"] is True
        assert "TRANSCRIPTION_FAILED" in result["transcription"]
        assert result["confidence"] == 0.0

    def test_mime_mapping(self):
        from services.audio_service import _mime_to_encoding
        assert _mime_to_encoding("audio/webm") == "WEBM_OPUS"
        assert _mime_to_encoding("audio/wav") == "LINEAR16"
        assert _mime_to_encoding("audio/ogg") == "OGG_OPUS"
        assert _mime_to_encoding("audio/flac") == "FLAC"
        assert _mime_to_encoding("audio/unknown") == "ENCODING_UNSPECIFIED"


# -----------------------------------------------------------------------------
# ai_service tests
# -----------------------------------------------------------------------------

class TestAIService:
    """Tests for services/ai_service.py - prompt engineering & validation logic."""

    VALID_METADATA = {
        "exif_found": True,
        "photo_datetime": "12-Aug-2026 09:23:41 IST",
        "hours_since_incident": 14.5,
        "within_72_hours": True,
        "gps_coordinates": {"lat": 30.1234, "lon": 75.5678},
    }

    def _valid_ai_payload(self, peril="Inundation", confidence=0.92, damage=65):
        return json.dumps({
            "crop_type": "Paddy",
            "peril_type": peril,
            "peril_confidence": confidence,
            "estimated_damage_percentage": damage,
            "damage_description": "Standing water visible across the paddy field.",
            "crop_growth_stage": "Grain Filling",
            "incident_date": "12-Aug-2026 09:23:41 IST",
            "requires_manual_review": False,
            "review_reason": None,
        })

    # --- _parse_and_validate unit tests -------------------------------------

    def test_valid_inundation_payload_passes(self):
        from services.ai_service import _parse_and_validate
        result = _parse_and_validate(self._valid_ai_payload("Inundation"), self.VALID_METADATA)
        assert result["peril_type"] == "Inundation"
        assert result["requires_manual_review"] is False
        assert result["crop_type"] == "Paddy"

    def test_all_valid_perils_accepted(self):
        from services.ai_service import _parse_and_validate, VALID_PERILS
        for peril in VALID_PERILS:
            result = _parse_and_validate(self._valid_ai_payload(peril), self.VALID_METADATA)
            assert result["peril_type"] == peril

    def test_invalid_peril_overridden_to_manual_review(self):
        from services.ai_service import _parse_and_validate
        payload = self._valid_ai_payload("Pest Attack")   # NOT a valid PMFBY localized peril
        result = _parse_and_validate(payload, self.VALID_METADATA)
        assert result["peril_type"] == "REQUIRES_MANUAL_REVIEW"
        assert result["requires_manual_review"] is True
        assert "Pest Attack" in result["review_reason"]

    def test_low_confidence_triggers_manual_review(self):
        from services.ai_service import _parse_and_validate
        payload = self._valid_ai_payload("Hailstorm", confidence=0.55)
        result = _parse_and_validate(payload, self.VALID_METADATA)
        assert result["requires_manual_review"] is True
        assert "confidence" in result["review_reason"].lower()

    def test_exactly_70_pct_confidence_does_not_trigger_review(self):
        from services.ai_service import _parse_and_validate
        payload = self._valid_ai_payload("Hailstorm", confidence=0.70)
        result = _parse_and_validate(payload, self.VALID_METADATA)
        # 0.70 == threshold, should NOT trigger review
        assert result["requires_manual_review"] is False

    def test_damage_percentage_clamped_to_100(self):
        from services.ai_service import _parse_and_validate
        payload = self._valid_ai_payload(damage=150)   # out-of-range
        result = _parse_and_validate(payload, self.VALID_METADATA)
        assert result["estimated_damage_percentage"] == 100

    def test_damage_percentage_clamped_to_0(self):
        from services.ai_service import _parse_and_validate
        payload = self._valid_ai_payload(damage=-10)
        result = _parse_and_validate(payload, self.VALID_METADATA)
        assert result["estimated_damage_percentage"] == 0

    def test_confidence_clamped_to_1(self):
        from services.ai_service import _parse_and_validate
        payload = self._valid_ai_payload(confidence=1.5)
        result = _parse_and_validate(payload, self.VALID_METADATA)
        assert result["peril_confidence"] <= 1.0

    def test_markdown_fences_stripped_before_parse(self):
        from services.ai_service import _parse_and_validate
        raw = "```json\n" + self._valid_ai_payload() + "\n```"
        result = _parse_and_validate(raw, self.VALID_METADATA)
        assert result["peril_type"] == "Inundation"

    def test_invalid_json_raises_value_error(self):
        from services.ai_service import _parse_and_validate
        with pytest.raises(ValueError, match="non-JSON"):
            _parse_and_validate("This is not JSON at all.", self.VALID_METADATA)

    def test_incident_date_injected_when_not_available(self):
        from services.ai_service import _parse_and_validate
        payload_dict = json.loads(self._valid_ai_payload())
        payload_dict["incident_date"] = "NOT_AVAILABLE"
        result = _parse_and_validate(json.dumps(payload_dict), self.VALID_METADATA)
        assert result["incident_date"] == self.VALID_METADATA["photo_datetime"]

    # --- analyze_claim fallback test ----------------------------------------

    @pytest.mark.asyncio
    async def test_analyze_claim_returns_mock_when_sdk_missing(self):
        from services.ai_service import analyze_claim
        with patch("services.ai_service._call_gemini", new_callable=AsyncMock,
                   side_effect=ImportError("vertexai not installed")):
            result = await analyze_claim(
                image_bytes=_make_plain_jpeg(),
                transcribed_text="Meri fasal doob gayi",
                metadata=self.VALID_METADATA,
            )
        assert result["crop_type"] is not None
        assert result["peril_type"] in {"Inundation", "REQUIRES_MANUAL_REVIEW"}

    @pytest.mark.asyncio
    async def test_analyze_claim_error_returns_safe_fallback(self):
        from services.ai_service import analyze_claim
        with patch("services.ai_service._call_gemini", new_callable=AsyncMock,
                   side_effect=Exception("API quota exceeded")):
            result = await analyze_claim(
                image_bytes=b"img",
                transcribed_text="test",
                metadata=self.VALID_METADATA,
            )
        assert result["requires_manual_review"] is True
        assert "API quota exceeded" in result["review_reason"]


# -----------------------------------------------------------------------------
# FastAPI endpoint integration test
# -----------------------------------------------------------------------------

class TestSubmitClaimEndpoint:
    """Integration tests for the /submit-claim FastAPI route."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def _mock_services(self):
        """Patch all three services to avoid any GCP calls."""
        meta_patch = patch(
            "main.extract_and_validate_metadata",
            return_value={
                "exif_found": True,
                "photo_datetime": "12-Aug-2026 09:23:41 IST",
                "hours_since_incident": 14.5,
                "within_72_hours": True,
                "gps_coordinates": {"lat": 30.12, "lon": 75.88},
                "raw_gps_string": "30°7'12.00\"N, 75°52'48.00\"E",
            },
        )
        audio_patch = patch(
            "main.transcribe_audio",
            new_callable=AsyncMock,
            return_value={
                "transcription": "Mere khet mein paani bhar gaya",
                "confidence": 0.92,
                "detected_language": "hi-IN",
                "is_fallback": False,
            },
        )
        ai_patch = patch(
            "main.analyze_claim",
            new_callable=AsyncMock,
            return_value={
                "crop_type": "Paddy",
                "peril_type": "Inundation",
                "peril_confidence": 0.91,
                "estimated_damage_percentage": 65,
                "damage_description": "Standing water visible.",
                "crop_growth_stage": "Grain Filling",
                "incident_date": "12-Aug-2026 09:23:41 IST",
                "requires_manual_review": False,
                "review_reason": None,
            },
        )
        return meta_patch, audio_patch, ai_patch

    def test_successful_claim_submission(self, client):
        meta_p, audio_p, ai_p = self._mock_services()
        with meta_p, audio_p, ai_p:
            jpeg = _make_plain_jpeg()
            response = client.post(
                "/submit-claim",
                data={"policy_number": "PMFBY-PB-2026-TEST01"},
                files={
                    "image": ("field.jpg", jpeg, "image/jpeg"),
                    "audio": ("note.webm", b"fake-audio-data", "audio/webm"),
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "INTAKE_SUCCESS"
        assert data["ncip_claim_payload"]["peril_type"] == "Inundation"
        assert data["ncip_claim_payload"]["policy_number"] == "PMFBY-PB-2026-TEST01"

    def test_missing_image_returns_422(self, client):
        response = client.post(
            "/submit-claim",
            data={"policy_number": "PMFBY-PB-2026-TEST01"},
            files={"audio": ("note.webm", b"data", "audio/webm")},
        )
        assert response.status_code == 422

    def test_missing_audio_returns_422(self, client):
        response = client.post(
            "/submit-claim",
            data={"policy_number": "PMFBY-PB-2026-TEST01"},
            files={"image": ("f.jpg", _make_plain_jpeg(), "image/jpeg")},
        )
        assert response.status_code == 422

    def test_empty_image_returns_400(self, client):
        meta_p, audio_p, ai_p = self._mock_services()
        with meta_p, audio_p, ai_p:
            response = client.post(
                "/submit-claim",
                data={"policy_number": "PMFBY-PB-2026-TEST01"},
                files={
                    "image": ("field.jpg", b"", "image/jpeg"),
                    "audio": ("note.webm", b"data", "audio/webm"),
                },
            )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_72_hour_violation_adds_warning(self, client):
        meta_p = patch(
            "main.extract_and_validate_metadata",
            return_value={
                "exif_found": True,
                "photo_datetime": "08-Aug-2026 09:00:00 IST",
                "hours_since_incident": 98.0,
                "within_72_hours": False,  # ← outside window
                "gps_coordinates": None,
                "raw_gps_string": None,
            },
        )
        audio_p = patch("main.transcribe_audio", new_callable=AsyncMock, return_value={
            "transcription": "test", "confidence": 0.9, "detected_language": "hi-IN", "is_fallback": False
        })
        ai_p = patch("main.analyze_claim", new_callable=AsyncMock, return_value={
            "crop_type": "Wheat", "peril_type": "Hailstorm", "peril_confidence": 0.85,
            "estimated_damage_percentage": 40, "damage_description": "Hail damage visible.",
            "crop_growth_stage": "Maturity", "incident_date": "08-Aug-2026 09:00:00 IST",
            "requires_manual_review": False, "review_reason": None,
        })
        with meta_p, audio_p, ai_p:
            r = client.post(
                "/submit-claim",
                data={"policy_number": "PMFBY-PB-2026-LATE01"},
                files={
                    "image": ("f.jpg", _make_plain_jpeg(), "image/jpeg"),
                    "audio": ("n.webm", b"audio", "audio/webm"),
                },
            )
        assert r.status_code == 200
        warnings = r.json()["warnings"]
        assert any("72-hour" in w or "CRITICAL" in w for w in warnings)
