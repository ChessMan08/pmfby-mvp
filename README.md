# PMFBY AI Intake Engine — Setup & Deployment Guide

> **Pradhan Mantri Fasal Bima Yojana** · Multimodal Crop Loss Claim Intake  
> Stack: FastAPI · Google Vertex AI (Gemini 1.5 Flash) · Google Cloud Speech-to-Text (Chirp v2) · Pillow

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Prerequisites](#2-prerequisites)
3. [Google Cloud Project Setup](#3-google-cloud-project-setup)
4. [Enable Required APIs](#4-enable-required-apis)
5. [Create a Service Account & Download Key](#5-create-a-service-account--download-key)
6. [Local Development Setup](#6-local-development-setup)
7. [Running the Server](#7-running-the-server)
8. [Testing the API](#8-testing-the-api)
9. [Deploying to Google Cloud Run](#9-deploying-to-google-cloud-run)
10. [Architecture Decisions & Engineering Notes](#10-architecture-decisions--engineering-notes)
11. [Known Limitations & Roadmap](#11-known-limitations--roadmap)

---

## 1. Project Structure

```
pmfby-mvp/
├── main.py                  # FastAPI app, routes, CORS, response assembly
├── index.html               # Single-page frontend (served by FastAPI)
├── services/
│   ├── __init__.py
│   ├── metadata_service.py  # EXIF extraction + 72-hour window validation
│   ├── audio_service.py     # Google Cloud Speech-to-Text (Chirp v2)
│   └── ai_service.py        # Vertex AI Gemini 1.5 Flash + prompt engineering
├── requirements.txt
├── Dockerfile               # Cloud Run-ready container
├── .env.example             # Environment variable template
├── .gitignore
└── README.md                # This file
```

---

## 2. Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | [python.org](https://python.org) |
| pip | 23+ | bundled with Python |
| Google Cloud SDK (`gcloud`) | Latest | [cloud.google.com/sdk](https://cloud.google.com/sdk/docs/install) |
| Docker (optional, for Cloud Run) | Latest | [docker.com](https://docker.com) |

Verify your tools:
```bash
python --version      # Python 3.11.x
gcloud --version      # Google Cloud SDK x.x.x
```

---

## 3. Google Cloud Project Setup

### 3a. Create a new project (skip if you already have one)

```bash
# Create the project
gcloud projects create pmfby-intake-mvp --name="PMFBY Intake Engine"

# Set it as the active project
gcloud config set project pmfby-intake-mvp
```

### 3b. Link a billing account (required for API usage)

```bash
# List your billing accounts
gcloud billing accounts list

# Link the billing account to your project
gcloud billing projects link pmfby-intake-mvp \
  --billing-account=XXXXXX-XXXXXX-XXXXXX
```

> **Cost estimate for MVP testing**: Running ~100 test claims will cost approximately  
> ₹15–40 (Gemini Flash @ ~$0.000075/image, Speech-to-Text @ ~$0.004/15s clip).

---

## 4. Enable Required APIs

Run this single command to enable all necessary APIs:

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  speech.googleapis.com \
  run.googleapis.com \
  containerregistry.googleapis.com \
  --project=pmfby-intake-mvp
```

Wait 2–3 minutes for propagation, then verify:

```bash
gcloud services list --enabled --project=pmfby-intake-mvp \
  --filter="name:(aiplatform OR speech OR run)"
```

---

## 5. Create a Service Account & Download Key

### 5a. Create the service account

```bash
gcloud iam service-accounts create pmfby-intake-sa \
  --display-name="PMFBY Intake Engine Service Account" \
  --project=pmfby-intake-mvp
```

### 5b. Grant the required IAM roles

```bash
PROJECT_ID="pmfby-intake-mvp"
SA_EMAIL="pmfby-intake-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Vertex AI (Gemini inference)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/aiplatform.user"

# Cloud Speech-to-Text
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/speech.client"
```

### 5c. Download the JSON key file

```bash
gcloud iam service-accounts keys create ./service-account-key.json \
  --iam-account="${SA_EMAIL}" \
  --project=pmfby-intake-mvp
```

> ⚠️ **Security**: The `service-account-key.json` file is already in `.gitignore`.  
> Never commit it to version control. Store it in a secrets manager (Secret Manager or  
> GitHub Secrets) for production deployments.

---

## 6. Local Development Setup

### 6a. Clone / enter the project directory

```bash
cd pmfby-mvp
```

### 6b. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 6c. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6d. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

```ini
GCP_PROJECT_ID=pmfby-intake-mvp
GCP_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=./service-account-key.json
APP_ENV=development
```

### 6e. Alternative: Authenticate via Application Default Credentials (ADC)

If you prefer not to use a key file during local development:

```bash
gcloud auth application-default login
# Follow browser OAuth flow

gcloud config set project pmfby-intake-mvp
```

Then set in `.env`:
```ini
# Leave GOOGLE_APPLICATION_CREDENTIALS blank — ADC will be used automatically
GCP_PROJECT_ID=pmfby-intake-mvp
```

---

## 7. Running the Server

### Development mode (with auto-reload)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production mode (multi-worker)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Open your browser at **http://localhost:8000**

You should see the PMFBY Fasal Suraksha intake interface.

---

## 8. Testing the API

### 8a. Health check

```bash
curl http://localhost:8000/health
# → {"status":"ok","service":"PMFBY AI Intake Engine"}
```

### 8b. Submit a test claim via cURL

```bash
curl -X POST http://localhost:8000/submit-claim \
  -F "policy_number=PMFBY-PB-2026-ABC123" \
  -F "image=@/path/to/test-field-photo.jpg" \
  -F "audio=@/path/to/test-voice-note.webm"
```

### 8c. Expected response structure

```json
{
  "status": "INTAKE_SUCCESS",
  "policy_details": { "farmer_name": "Rajesh Kumar", "insured_crop": "Paddy (Kharif)", ... },
  "metadata_analysis": {
    "exif_found": true,
    "photo_datetime": "12-Aug-2026 09:23:41 IST",
    "hours_since_incident": 14.5,
    "within_72_hours": true,
    "gps_coordinates": { "lat": 30.234567, "lon": 75.876543 }
  },
  "transcription": {
    "transcription": "Kal raat bahut tez baarish hui...",
    "confidence": 0.94,
    "detected_language": "hi-IN",
    "is_fallback": false
  },
  "ncip_claim_payload": {
    "crop_type": "Paddy",
    "peril_type": "Inundation",
    "peril_confidence": 0.91,
    "estimated_damage_percentage": 65,
    "damage_description": "Standing water visible across approximately 65% of the paddy field...",
    "crop_growth_stage": "Grain Filling",
    "incident_date": "12-Aug-2026 09:23:41 IST",
    "within_72_hours": true,
    "policy_number": "PMFBY-PB-2026-ABC123",
    "submission_channel": "AI_INTAKE_ENGINE_v1"
  },
  "warnings": []
}
```

### 8d. Running unit tests

```bash
pytest tests/ -v
```

---

## 9. Deploying to Google Cloud Run

### 9a. Build and push the Docker image

```bash
PROJECT_ID="pmfby-intake-mvp"

# Configure Docker to use gcloud credentials
gcloud auth configure-docker

# Build the image
docker build -t gcr.io/${PROJECT_ID}/pmfby-intake:v1 .

# Push to Google Container Registry
docker push gcr.io/${PROJECT_ID}/pmfby-intake:v1
```

### 9b. Deploy to Cloud Run

```bash
gcloud run deploy pmfby-intake \
  --image gcr.io/${PROJECT_ID}/pmfby-intake:v1 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --service-account "pmfby-intake-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --set-env-vars "GCP_PROJECT_ID=${PROJECT_ID},GCP_LOCATION=us-central1,APP_ENV=production" \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 10 \
  --concurrency 80 \
  --timeout 120
```

> **Note on `--allow-unauthenticated`**: For production, remove this flag and implement  
> Firebase Authentication or IAP (Identity-Aware Proxy) to authenticate farmers via  
> mobile OTP before claim submission.

### 9c. Store secrets securely (production best practice)

Instead of environment variables, use Google Secret Manager:

```bash
# Store the service account key as a secret (if not using workload identity)
echo -n "$(cat service-account-key.json)" | \
  gcloud secrets create pmfby-sa-key --data-file=-

# Grant Cloud Run access to the secret
gcloud secrets add-iam-policy-binding pmfby-sa-key \
  --member="serviceAccount:pmfby-intake-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

---

## 10. Architecture Decisions & Engineering Notes

### Why Gemini 1.5 Flash over Pro?

Flash provides ~10x lower latency and ~4x lower cost vs Pro, which is critical when a
farmer in a rural area with poor connectivity needs a response within seconds. The
constrained JSON output schema and closed-set peril taxonomy mean we don't need Pro's
extended reasoning — Flash's speed optimises for the 72-hour window urgency.

### Why `response_mime_type: application/json` in the Gemini call?

Gemini's JSON mode (available in Flash 002+) forces the model to output structurally
valid JSON without markdown fences, eliminating the need for fragile regex stripping.
The post-processing validator in `_parse_and_validate()` acts as a secondary safety net.

### Why Chirp v2 over Standard/Enhanced Speech models?

Chirp v2 is Google's most recent universal speech model, purpose-built for:
- Low-resource language support (crucial for dialects like Chhattisgarhi, Bhojpuri)
- Code-mixed input (Hindi + English agricultural terms like "field mein waterlogging")
- High background noise robustness (wind, rain, livestock during recording)

The `AutoDetectDecodingConfig` removes the need to specify audio encoding format,
accommodating the varied audio codecs produced by different Android devices.

### Why EXIF over server-side timestamp?

A farmer could submit a claim for damage that occurred 60 hours ago. Using the server's
`datetime.now()` would mark it as 0 hours elapsed — defeating the 72-hour check. EXIF
`DateTimeOriginal` captures when the shutter fired, giving a tamper-evident timestamp.
The graceful degradation path (no EXIF → warning flag, not rejection) prevents the AI
from becoming a barrier rather than an enabler.

### Hallucination prevention strategy

Three layers of defence against AI hallucination:
1. **Closed-set enum** — `peril_type` must be one of 5 valid values or `REQUIRES_MANUAL_REVIEW`
2. **Confidence gating** — sub-70% confidence forces manual review regardless of peril output
3. **Schema validation** — `_parse_and_validate()` rejects or overrides any out-of-spec value

The system never rejects a claim — it escalates ambiguous cases to human assessors,
ensuring no farmer is silently denied due to an AI error.

---

## 11. Known Limitations & Roadmap

| Limitation | Current Behaviour | Roadmap Fix |
|------------|-------------------|-------------|
| WhatsApp photo EXIF stripping | Warning displayed to user | Deep link to in-app camera capture |
| Single photo analysis | Only one image per claim | Multi-image upload with composite damage scoring |
| Mock policy fetch | Returns hardcoded policy data | Integrate real PMFBY/NCIP farmer lookup API |
| No OTP authentication | Policy number is self-declared | Firebase Phone Auth with Aadhaar-linked OTP |
| English-only UI labels | Hindi/regional labels only partially implemented | Full i18n via `react-intl` or `gettext` |
| Pest damage misclassification | Flagged for manual review | Dedicated pest vs. calamity binary classifier |
| No offline support | Requires live internet during submission | PWA with IndexedDB queue + background sync |
| Audio length limit | 60-second auto-stop | Configurable; increase for complex descriptions |

---

## Support

For issues with GCP API setup, refer to:
- [Vertex AI Gemini Documentation](https://cloud.google.com/vertex-ai/generative-ai/docs)
- [Cloud Speech-to-Text v2 (Chirp)](https://cloud.google.com/speech-to-text/v2/docs/chirp-model)
- [PMFBY Operational Guidelines](https://pmfby.gov.in)
