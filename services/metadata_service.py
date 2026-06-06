import io
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

logger = logging.getLogger(__name__)

# PMFBY mandated window (hours)
REPORTING_WINDOW_HOURS = 72

# EXIF tag IDs we care about
_EXIF_TAG_MAP = {v: k for k, v in TAGS.items()}
_DATETIME_ORIGINAL_TAG = _EXIF_TAG_MAP.get("DateTimeOriginal", 36867)
_GPS_INFO_TAG = _EXIF_TAG_MAP.get("GPSInfo", 34853)


def extract_and_validate_metadata(image_bytes: bytes) -> dict[str, Any]:

    result: dict[str, Any] = {
        "exif_found": False,
        "photo_datetime": None,
        "hours_since_incident": None,
        "within_72_hours": False,
        "gps_coordinates": None,
        "raw_gps_string": None,
    }

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        logger.warning("Could not open image for EXIF extraction: %s", exc)
        return result

    # Pillow exposes raw EXIF only for JPEG; for PNG/WEBP it may return None
    exif_data = img._getexif() if hasattr(img, "_getexif") else None  # type: ignore[attr-defined]

    if not exif_data:
        logger.info("No EXIF data found in uploaded image.")
        return result

    result["exif_found"] = True
    tag_lookup = {TAGS.get(tag, tag): value for tag, value in exif_data.items()}

    # -----------------------------------------------------------------------
    # 1. DateTimeOriginal → 72-hour check
    # -----------------------------------------------------------------------
    dt_str: str | None = tag_lookup.get("DateTimeOriginal")
    if dt_str:
        try:
            # EXIF format: "YYYY:MM:DD HH:MM:SS"
            photo_dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
            # Treat as IST (UTC+5:30) - in production, respect device timezone
            ist_offset = timedelta(hours=5, minutes=30)
            photo_dt_utc = photo_dt - ist_offset
            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

            delta = now_utc - photo_dt_utc
            hours_elapsed = delta.total_seconds() / 3600

            result["photo_datetime"] = photo_dt.strftime("%d-%b-%Y %H:%M:%S IST")
            result["hours_since_incident"] = round(hours_elapsed, 2)
            result["within_72_hours"] = 0 <= hours_elapsed <= REPORTING_WINDOW_HOURS

            logger.info(
                "Photo taken %.1f hours ago. Within 72h window: %s",
                hours_elapsed,
                result["within_72_hours"],
            )
        except ValueError as exc:
            logger.warning("Could not parse EXIF DateTimeOriginal '%s': %s", dt_str, exc)

    # -----------------------------------------------------------------------
    # 2. GPS coordinates
    # -----------------------------------------------------------------------
    gps_raw = exif_data.get(_GPS_INFO_TAG)
    if gps_raw:
        try:
            gps_decoded = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
            lat = _dms_to_decimal(
                gps_decoded.get("GPSLatitude"),
                gps_decoded.get("GPSLatitudeRef", "N"),
            )
            lon = _dms_to_decimal(
                gps_decoded.get("GPSLongitude"),
                gps_decoded.get("GPSLongitudeRef", "E"),
            )
            if lat is not None and lon is not None:
                result["gps_coordinates"] = {"lat": round(lat, 6), "lon": round(lon, 6)}
                result["raw_gps_string"] = (
                    f"{_format_dms(gps_decoded.get('GPSLatitude'))} "
                    f"{gps_decoded.get('GPSLatitudeRef', 'N')}, "
                    f"{_format_dms(gps_decoded.get('GPSLongitude'))} "
                    f"{gps_decoded.get('GPSLongitudeRef', 'E')}"
                )
                logger.info("GPS extracted: lat=%.6f, lon=%.6f", lat, lon)
        except Exception as exc:
            logger.warning("GPS extraction failed: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _dms_to_decimal(dms, ref: str) -> float | None:
    """Convert degrees/minutes/seconds tuple to signed decimal degrees."""
    if dms is None:
        return None
    try:
        degrees = float(dms[0])
        minutes = float(dms[1])
        seconds = float(dms[2])
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if ref in ("S", "W"):
            decimal = -decimal
        return decimal
    except (IndexError, TypeError, ZeroDivisionError):
        return None


def _format_dms(dms) -> str:
    """Format a DMS tuple as a human-readable string."""
    if dms is None:
        return "?"
    try:
        d = float(dms[0])
        m = float(dms[1])
        s = float(dms[2])
        return f"{d:.0f}°{m:.0f}'{s:.2f}\""
    except Exception:
        return str(dms)
