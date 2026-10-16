import asyncio
import base64
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import io
import joblib
import httpx

from agri_agent import AgentAPIError, AgentConfigurationError, run_agri_agent
from database import (
    SessionLocal,
    CropRecord,
    LeafRecord,
    IrrigationRecord,
    MarketRecord,
    ChatbotRecord,
    init_db,
)

app = FastAPI(
    title="Smart AgriGuide Core API",
    version="2.8.0",
    description="Full-stack AI Engine: Real-Time Satellite Telemetry, Live Mandi Feeds, Multi-Plant Pathology Vision Triage & SQL Persistence"
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

def get_db():
    if SessionLocal is None:
        return None
    try:
        db = SessionLocal()
        return db
    except Exception:
        return None

# Load Machine Learning Model Artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_recommender.joblib")
if os.path.exists(MODEL_PATH):
    crop_model = joblib.load(MODEL_PATH)
else:
    crop_model = None

# -------------------------------------------------------------
# LIVE DATASET ENDPOINT 1: Real-Time Meteorological Telemetry
# -------------------------------------------------------------
@app.get("/api/v1/live-weather-telemetry")
async def get_live_weather_telemetry(lat: float = Query(13.0827), lon: float = Query(80.2707)):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,surface_pressure&daily=precipitation_sum&timezone=auto"
    try:
        async with httpx.AsyncClient(timeout=4.5) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                current = data.get("current", {})
                daily = data.get("daily", {})
                
                temp = float(current.get("temperature_2m", 25.0))
                humidity = float(current.get("relative_humidity_2m", 78.0))
                rain_today = float(daily.get("precipitation_sum", [180.0])[0]) if daily.get("precipitation_sum") else 180.0
                
                estimated_rainfall = max(rain_today * 30.0, 185.0) if rain_today <= 5.0 else min(rain_today * 25.0, 320.0)

                return {
                    "status": "success",
                    "source": "Open-Meteo Satellite Feed (Live)",
                    "coordinates": {"latitude": lat, "longitude": lon},
                    "telemetry": {
                        "temperature": round(temp, 1),
                        "humidity": round(humidity, 1),
                        "rainfall": round(estimated_rainfall, 1),
                        "surface_pressure_hpa": current.get("surface_pressure", 1010.5)
                    }
                }
    except Exception as e:
        print(f"Weather API error: {e}")
    
    return {
        "status": "fallback",
        "source": "Regional Agro-Climatic Baseline",
        "coordinates": {"latitude": lat, "longitude": lon},
        "telemetry": {
            "temperature": 25.0,
            "humidity": 80.0,
            "rainfall": 180.0,
            "surface_pressure_hpa": 1013.2
        }
    }

# -------------------------------------------------------------
# AGMARKNET daily mandi prices via the data.gov.in resource API.
# The source is updated daily; it is not a tick-by-tick exchange feed.
# -------------------------------------------------------------
AGMARKNET_RESOURCE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
MARKET_FEED_CACHE_SECONDS = 300
MARKET_FEED_RECORD_LIMIT = 100
MARKET_FEED_DISTRICT_RECORD_LIMIT = 10
MARKET_FEED_TIMEOUT_SECONDS = 20.0
_market_feed_cache: dict[str, tuple[float, list[dict]]] = {}
MARKET_COMMODITY_ALIASES = {
    "Rice": {"rice", "paddy", "paddy(dhan)(common)", "paddy(dhan)(grade a)"},
    "Tomato": {"tomato"},
    "Maize": {"maize"},
    "Cotton": {"cotton", "cotton(kapas)"},
    "Potato": {"potato"},
    "Chickpea": {"chickpea", "gram", "bengal gram", "bengal gram(gram)(whole)", "chana"},
}


async def fetch_agmarknet_state_records(state: str, district: str | None = None) -> tuple[list[dict], str | None]:
    """Fetch recent state or district records and cache them briefly."""
    api_key = os.getenv("DATA_GOV_IN_API_KEY", "").strip()
    if not api_key:
        return [], "missing_api_key"

    normalized_district = district.strip() if district and district.strip() else None
    cache_key = f"{state.casefold().strip()}|{(normalized_district or '').casefold()}"
    cached = _market_feed_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < MARKET_FEED_CACHE_SECONDS:
        return cached[1], None

    params = {
        "api-key": api_key,
        "format": "json",
        # This resource's API schema exposes State as the keyword filter.
        "filters[state.keyword]": state,
        # Keep the payload small; the UI only needs a recent sample to derive
        # current crop prices, and the upstream endpoint can be slow on large pages.
        "limit": MARKET_FEED_DISTRICT_RECORD_LIMIT if normalized_district else MARKET_FEED_RECORD_LIMIT,
    }
    if normalized_district:
        params["filters[district]"] = normalized_district
    for attempt in range(2):
        try:
            def request_payload() -> dict:
                query = urllib.parse.urlencode(params)
                request_url = f"{AGMARKNET_RESOURCE_URL}?{query}"
                request = urllib.request.Request(
                    request_url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
                        "Accept": "application/json",
                    },
                )
                with urllib.request.urlopen(request, timeout=MARKET_FEED_TIMEOUT_SECONDS) as response:
                    return json.loads(response.read().decode("utf-8"))

            # urllib picks up the Windows system proxy when one is configured;
            # run its blocking request off the async event loop.
            payload = await asyncio.to_thread(request_payload)
            if not isinstance(payload, dict) or payload.get("error"):
                return [], "feed_unavailable"
            records = payload.get("records", [])
            if not isinstance(records, list):
                return [], "invalid_response"
            _market_feed_cache[cache_key] = (time.monotonic(), records)
            return records, None
        except urllib.error.HTTPError as exc:
            # Log only the status code: the exception URL can include the API key.
            print(f"AGMARKNET data.gov.in request failed (HTTP {exc.code})")
            return [], "feed_unavailable"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == 0:
                await asyncio.sleep(0.5)
                continue
            # Do not print the exception string; it may contain the API key URL.
            print(f"AGMARKNET data.gov.in request failed ({type(exc).__name__})")
            return [], "feed_unavailable"
        except (ValueError, TypeError) as exc:
            # Do not print the exception string: it can include the API key in the request URL.
            print(f"AGMARKNET data.gov.in request failed ({type(exc).__name__})")
            return [], "feed_unavailable"

    return [], "feed_unavailable"


def parse_market_date(value: str | None) -> datetime:
    if not value:
        return datetime.min
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        for date_format in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, date_format)
            except ValueError:
                continue
    return datetime.min


def market_price_number(value) -> float | None:
    try:
        number = float(str(value).replace(",", "").strip())
        return number if number > 0 else None
    except (TypeError, ValueError):
        return None


def matching_market_records(records: list[dict], crop_name: str) -> list[dict]:
    aliases = {alias.casefold() for alias in MARKET_COMMODITY_ALIASES.get(crop_name, {crop_name.casefold()})}
    return [
        record for record in records
        if any(
            str(record.get("commodity", "")).strip().casefold() == alias
            or str(record.get("commodity", "")).strip().casefold().startswith(alias + "(")
            for alias in aliases
        )
        and market_price_number(record.get("modal_price")) is not None
    ]


@app.get("/api/v1/live-mandi-feed")
async def get_live_mandi_feed(
    state: str = Query(default="Andhra Pradesh", min_length=2, max_length=80),
    district: str | None = Query(default=None, max_length=80),
):
    records, error = await fetch_agmarknet_state_records(state, district)
    if error:
        message = (
            "Configure DATA_GOV_IN_API_KEY on the backend to load official daily mandi prices."
            if error == "missing_api_key"
            else "The official market feed could not be reached. Please try again later."
        )
        return {"status": error, "live_data_available": False, "source": "AGMARKNET via data.gov.in", "state": state, "district": district, "message": message, "records": []}

    latest_records = []
    for crop_name in MARKET_COMMODITY_ALIASES:
        matches = matching_market_records(records, crop_name)
        if not matches:
            continue
        latest = max(matches, key=lambda row: parse_market_date(row.get("arrival_date")))
        latest_records.append({
            "commodity": crop_name,
            "market_commodity": latest.get("commodity"),
            "market": latest.get("market"),
            "district": latest.get("district"),
            "state": latest.get("state", state),
            "min_price": market_price_number(latest.get("min_price")),
            "max_price": market_price_number(latest.get("max_price")),
            "modal_price": market_price_number(latest.get("modal_price")),
            "arrival_date": latest.get("arrival_date"),
        })
    latest_records.sort(key=lambda row: parse_market_date(row.get("arrival_date")), reverse=True)
    return {
        "status": "success" if latest_records else "no_data",
        "live_data_available": bool(latest_records),
        "source": "AGMARKNET via data.gov.in",
        "state": state,
        "district": district,
        "message": None if latest_records else f"No matching mandi prices were returned for {state}.",
        "records": latest_records,
    }

# -------------------------------------------------------------
# COMPREHENSIVE MULTI-PLANT PATHOLOGY CATALOG (15+ DISEASES)
# -------------------------------------------------------------
DISEASE_CATALOG = {
    # Lemon / Citrus
    "Lemon - Citrus Leaf Miner Damage": {
        "crop": "Lemon (Citrus)",
        "pathogen": "Phyllocnistis citrella (Larval Pest)",
        "severity": "Moderate to High",
        "remedy": "Apply Neem oil spray (5ml/L) or Spinosad/Abamectin during new vegetative flush. Prune severely curled shoots to disrupt pest breeding cycle."
    },
    "Lemon - Citrus Canker": {
        "crop": "Lemon (Citrus)",
        "pathogen": "Xanthomonas axonopodis pv. citri (Bacterium)",
        "severity": "High",
        "remedy": "Spray Copper Oxychloride (3g/L) or Streptocycline (1g/10L) combined with Bordeaux mixture. Prune and destroy infected canker twigs."
    },
    "Lemon - Healthy Foliage": {
        "crop": "Lemon (Citrus)",
        "pathogen": "None (Healthy Tissue)",
        "severity": "Nominal",
        "remedy": "Maintain regular micronutrient spray (Zinc Sulphate 0.5% + Ferrous Sulphate 0.2%) and uniform drip moisture."
    },
    # Tomato
    "Tomato - Early Blight": {
        "crop": "Tomato",
        "pathogen": "Alternaria solani (Fungus)",
        "severity": "Moderate",
        "remedy": "Apply Mancozeb 75% WP (2.5g/L) or Chlorothalonil foliar spray every 7-10 days. Prune lower diseased foliage and avoid overhead sprinkling."
    },
    "Tomato - Late Blight": {
        "crop": "Tomato",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "severity": "Critical",
        "remedy": "Immediately isolate affected plot sectors. Apply systemic fungicides like Metalaxyl-M (2g/L) or Cymoxanil + Mancozeb."
    },
    "Tomato - Yellow Leaf Curl Virus": {
        "crop": "Tomato",
        "pathogen": "Tomato Yellow Leaf Curl Begomovirus (TYLCV)",
        "severity": "High",
        "remedy": "Control whitefly vectors using yellow sticky traps and spray Imidacloprid (0.3ml/L) or Thiamethoxam. Remove heavily stunted plants."
    },
    # Potato
    "Potato - Early Blight": {
        "crop": "Potato",
        "pathogen": "Alternaria solani (Fungus)",
        "severity": "Moderate",
        "remedy": "Implement 3-year crop rotation. Apply Chlorothalonil or Mancozeb before canopy closure to prevent concentric target lesions."
    },
    "Potato - Late Blight": {
        "crop": "Potato",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "severity": "Critical",
        "remedy": "Spray Dimethomorph (1g/L) or Cymoxanil immediately upon lesion detection. Destroy infected tuber foliage before harvest."
    },
    # Corn (Maize)
    "Corn (Maize) - Common Rust": {
        "crop": "Corn (Maize)",
        "pathogen": "Puccinia sorghi (Fungus)",
        "severity": "Low to Moderate",
        "remedy": "Apply Azoxystrobin or Pyraclostrobin foliar spray if golden-brown rust pustules emerge prior to tassel/silking stage."
    },
    "Corn (Maize) - Northern Leaf Blight": {
        "crop": "Corn (Maize)",
        "pathogen": "Exserohilum turcicum (Fungus)",
        "severity": "Moderate",
        "remedy": "Apply Propiconazole or Azoxystrobin. Practice deep tillage to bury infected crop residue from previous harvest."
    },
    # Rice (Paddy)
    "Rice - Bacterial Leaf Blight": {
        "crop": "Rice (Paddy)",
        "pathogen": "Xanthomonas oryzae pv. oryzae",
        "severity": "High",
        "remedy": "Drain excess water from field for 3-4 days. Avoid excess nitrogen fertilization; spray Copper Hydroxide (2g/L) + Streptocycline."
    },
    "Rice - Brown Spot Disease": {
        "crop": "Rice (Paddy)",
        "pathogen": "Bipolaris oryzae (Fungus)",
        "severity": "Moderate",
        "remedy": "Correct soil potassium deficiency. Spray Tricyclazole (0.6g/L) or Carbendazim at tillering and panicle emergence."
    },
    # Cotton
    "Cotton - Bacterial Blight (Angular Leaf Spot)": {
        "crop": "Cotton",
        "pathogen": "Xanthomonas citri pv. malvacearum",
        "severity": "Moderate to High",
        "remedy": "Spray Copper Oxychloride (3g/L) combined with Streptomycin sulphate (1g/10L). Avoid dense plant spacing to facilitate canopy ventilation."
    },
    # Apple
    "Apple - Apple Scab": {
        "crop": "Apple",
        "pathogen": "Venturia inaequalis (Fungus)",
        "severity": "High",
        "remedy": "Apply Captan 50 WP or Myclobutanil at green tip through petal fall periods. Collect and burn fallen leaves in winter."
    },
    # General / Healthy
    "Plant - Healthy Specimen": {
        "crop": "All Species",
        "pathogen": "None (Healthy Tissue)",
        "severity": "Nominal",
        "remedy": "Maintain balanced N-P-K nutrient schedules, regular scout monitoring, and optimal drip irrigation moisture."
    }
}

class CropFeatures(BaseModel):
    nitrogen: float = Field(..., ge=0, le=300, example=90.0)
    phosphorus: float = Field(..., ge=0, le=300, example=42.0)
    potassium: float = Field(..., ge=0, le=300, example=43.0)
    temperature: float = Field(..., ge=-10, le=60, example=24.5)
    humidity: float = Field(..., ge=0, le=100, example=82.0)
    ph: float = Field(..., ge=0, le=14, example=6.5)
    rainfall: float = Field(..., ge=0, le=1000, example=202.0)

class IrrigationTelemetry(BaseModel):
    soil_moisture_pct: float = Field(..., ge=0, le=100, example=22.0)
    field_capacity_pct: float = Field(default=35.0, ge=10, le=50)
    root_zone_depth_mm: float = Field(default=300.0, ge=50, le=1500)
    latitude: float = Field(default=13.0827, example=13.0827)
    longitude: float = Field(default=80.2707, example=80.2707)

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": crop_model is not None,
        "agent_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "agent_model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    }

# -------------------------------------------------------------
# MODULE 1: Crop Suitability Recommendation
# -------------------------------------------------------------
@app.post("/api/v1/recommend-crop")
def recommend_crop(data: CropFeatures):
    if not crop_model:
        raise HTTPException(status_code=500, detail="Crop recommendation model artifact not found.")

    features = np.array([[
        data.nitrogen, data.phosphorus, data.potassium,
        data.temperature, data.humidity, data.ph, data.rainfall
    ]])
    prediction = crop_model.predict(features)[0]
    probabilities = crop_model.predict_proba(features)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]

    top_crops = [
        {"crop": crop_model.classes_[idx].title(), "confidence": round(float(probabilities[idx]) * 100, 2)}
        for idx in top_indices
    ]
    top_confidence = top_crops[0]["confidence"] if top_crops else 0.0

    db = get_db()
    if db:
        try:
            record = CropRecord(
                nitrogen=data.nitrogen,
                phosphorus=data.phosphorus,
                potassium=data.potassium,
                temperature=data.temperature,
                humidity=data.humidity,
                ph=data.ph,
                rainfall=data.rainfall,
                predicted_crop=prediction.title(),
                confidence=top_confidence
            )
            db.add(record)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"SQL Insert Error (CropRecord): {e}")
        finally:
            db.close()

    return {
        "primary_recommendation": prediction.title(),
        "top_recommendations": top_crops,
        "input_summary": data.dict()
    }

# -------------------------------------------------------------
# MODULE 2: Gemini image-based plant and leaf screening
# -------------------------------------------------------------
@app.post("/api/v1/diagnose-leaf")
async def diagnose_leaf(file: UploadFile = File(...), plant_type: str = Form(default="auto")):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image format.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Choose an image smaller than 10 MB.")

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img.thumbnail((1600, 1600))
        normalized_image = io.BytesIO()
        img.save(normalized_image, format="JPEG", quality=88, optimize=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted image file.")

    chosen_plant = plant_type.lower().strip()
    plant_labels = {
        "lemon": "Lemon / citrus",
        "tomato": "Tomato",
        "rice": "Rice / paddy",
        "potato": "Potato",
        "corn": "Corn / maize",
        "cotton": "Cotton",
        "apple": "Apple",
    }
    if chosen_plant not in {"auto", *plant_labels.keys()}:
        raise HTTPException(status_code=400, detail="Choose Auto-detect or a supported plant from the list.")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Leaf photo analysis is not configured. Set GEMINI_API_KEY in the backend environment, then restart FastAPI.",
        )

    selected_hint = (
        f"The user selected {plant_labels[chosen_plant]} as a hint. Check whether the photo supports that choice; do not assume it is correct."
        if chosen_plant != "auto"
        else "The user requested automatic plant identification from the image."
    )
    prompt = f"""Analyze this uploaded plant-leaf photo for a farmer in India.
{selected_hint}
Identify the plant only as specifically as the visible leaf supports. If uncertain, say Uncertain and use low confidence. Assess only visible symptoms; do not invent disease, pathogen, or severity. Do not claim 100% certainty and do not provide pesticide names or chemical doses. If the photo is not a clear leaf image, say so and recommend a clearer photo.
Return a concise, cautious visual screening result in the required JSON format. Confidence must be high, medium, or low and is only a model estimate, not a measured accuracy score."""
    vision_schema = {
        "type": "object",
        "properties": {
            "plant_name": {"type": "string"},
            "scientific_name": {"type": "string"},
            "plant_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "health_status": {"type": "string", "enum": ["healthy-looking", "possible issue", "uncertain"]},
            "diagnosis": {"type": "string"},
            "disease_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "severity": {"type": "string", "enum": ["none", "mild", "moderate", "severe", "uncertain"]},
            "suspected_cause": {"type": "string"},
            "visible_symptoms": {"type": "array", "items": {"type": "string"}},
            "recommended_next_steps": {"type": "string"},
            "image_quality_note": {"type": "string"},
            "requires_local_confirmation": {"type": "boolean"},
        },
        "required": [
            "plant_name", "scientific_name", "plant_confidence", "health_status",
            "diagnosis", "disease_confidence", "severity", "suspected_cause",
            "visible_symptoms", "recommended_next_steps", "image_quality_note",
            "requires_local_confirmation",
        ],
    }
    image_data = base64.b64encode(normalized_image.getvalue()).decode("ascii")
    payload = {
        "model": os.getenv("GEMINI_VISION_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")),
        "input": [
            {"type": "text", "text": prompt},
            {"type": "image", "data": image_data, "mime_type": "image/jpeg"},
        ],
        "response_format": {"type": "text", "mime_type": "application/json", "schema": vision_schema},
        "store": False,
        "generation_config": {"thinking_level": "minimal"},
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=10.0)) as client:
            response = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/interactions",
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Gemini leaf photo analysis timed out. Try again with a clear photo.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Could not connect to Gemini for leaf photo analysis.") from exc

    if response.status_code in {401, 403}:
        raise HTTPException(status_code=502, detail="Gemini rejected the API key. Check GEMINI_API_KEY in the backend environment.")
    if response.status_code in {429, 503}:
        raise HTTPException(status_code=503, detail="Gemini image analysis is busy or rate-limited. Please try again shortly.")
    if not response.is_success:
        raise HTTPException(status_code=502, detail=f"Gemini image analysis failed ({response.status_code}).")

    try:
        interaction = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Gemini returned an invalid image-analysis response.") from exc
    output_text = interaction.get("output_text", "") if isinstance(interaction, dict) else ""
    if not isinstance(output_text, str) or not output_text.strip():
        output_parts = []
        for step in interaction.get("steps", []) if isinstance(interaction, dict) else []:
            if not isinstance(step, dict) or step.get("type") not in {"model_output", "text"}:
                continue
            content = step.get("content", [])
            if isinstance(content, str):
                output_parts.append(content)
            elif isinstance(content, list):
                output_parts.extend(
                    item["text"] for item in content
                    if isinstance(item, dict) and isinstance(item.get("text"), str)
                )
        output_text = "\n".join(output_parts).strip()
    try:
        result = json.loads(output_text)
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object.")
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Gemini could not return a usable leaf analysis. Please try another photo.") from exc

    allowed_confidence = {"high", "medium", "low"}
    plant_confidence = str(result.get("plant_confidence", "low")).lower()
    disease_confidence = str(result.get("disease_confidence", "low")).lower()
    if plant_confidence not in allowed_confidence:
        plant_confidence = "low"
    if disease_confidence not in allowed_confidence:
        disease_confidence = "low"
    plant_name = str(result.get("plant_name", "Uncertain")).strip()[:120] or "Uncertain"
    scientific_name = str(result.get("scientific_name", "")).strip()[:120]
    diagnosis = str(result.get("diagnosis", "Unable to assess from this photo.")).strip()[:300]
    health_status = str(result.get("health_status", "uncertain")).strip()[:80]
    severity = str(result.get("severity", "uncertain")).strip()[:50]
    suspected_cause = str(result.get("suspected_cause", "Uncertain")).strip()[:150]
    raw_symptoms = result.get("visible_symptoms", [])
    if not isinstance(raw_symptoms, list):
        raw_symptoms = []
    symptoms = [str(item).strip()[:180] for item in raw_symptoms if str(item).strip()][:8]
    next_steps = str(result.get("recommended_next_steps", "Take a clear photo of one leaf and confirm the result with a local agricultural extension officer.")).strip()[:1200]
    image_quality_note = str(result.get("image_quality_note", "")).strip()[:300]
    requires_local_confirmation = bool(result.get("requires_local_confirmation", True))

    db = get_db()
    if db:
        try:
            record = LeafRecord(
                diagnosis=diagnosis[:150],
                pathogen_type=suspected_cause[:150],
                severity=severity,
                confidence_score=None,
                recommended_treatment=next_steps
            )
            db.add(record)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"SQL Insert Error (LeafRecord): {e}")
        finally:
            db.close()

    return {
        "plant_name": plant_name,
        "scientific_name": scientific_name,
        "plant_confidence": plant_confidence,
        "health_status": health_status,
        "diagnosis": diagnosis,
        "disease_confidence": disease_confidence,
        "severity": severity,
        "suspected_cause": suspected_cause,
        "visible_symptoms": symptoms,
        "recommended_next_steps": next_steps,
        "image_quality_note": image_quality_note,
        "requires_local_confirmation": requires_local_confirmation,
        "model": payload["model"],
        "result_type": "ai_visual_estimate_not_guaranteed",
    }

# -------------------------------------------------------------
# MODULE 3: Smart Hydrological Telemetry
# -------------------------------------------------------------
@app.post("/api/v1/advisory-irrigation")
async def calculate_irrigation(telemetry: IrrigationTelemetry):
    prob_rain = 0.20
    forecast_rain_mm = 0.0
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={telemetry.latitude}&longitude={telemetry.longitude}&daily=precipitation_probability_max,precipitation_sum&timezone=auto"
        async with httpx.AsyncClient() as client:
            res = await client.get(url, timeout=4.0)
            if res.status_code == 200:
                data = res.json()
                prob_rain = data["daily"]["precipitation_probability_max"][0] / 100.0
                forecast_rain_mm = float(data["daily"]["precipitation_sum"][0])
    except Exception:
        prob_rain = 0.15
        forecast_rain_mm = 0.0

    theta_s = telemetry.soil_moisture_pct
    theta_fc = telemetry.field_capacity_pct
    zr = telemetry.root_zone_depth_mm

    if prob_rain >= 0.65 or theta_s >= (0.85 * theta_fc):
        irrigation_vol_mm = 0.0
        status = "Irrigation Suppressed: Adequate soil moisture or impending high-probability rainfall."
    else:
        deficit_depth = max(0.0, (theta_fc - theta_s) / 100.0 * zr)
        irrigation_vol_mm = round(max(0.0, deficit_depth - (prob_rain * forecast_rain_mm)), 2)
        status = f"Discharge {irrigation_vol_mm} mm ({irrigation_vol_mm * 10:.1f} m³/hectare) to restore field capacity."

    db = get_db()
    if db:
        try:
            record = IrrigationRecord(
                soil_moisture_pct=theta_s,
                field_capacity_pct=theta_fc,
                precipitation_probability_pct=round(prob_rain * 100, 1),
                forecast_rainfall_mm=forecast_rain_mm,
                recommended_irrigation_depth_mm=irrigation_vol_mm,
                advisory_status=status
            )
            db.add(record)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"SQL Insert Error (IrrigationRecord): {e}")
        finally:
            db.close()

    return {
        "precipitation_probability_pct": round(prob_rain * 100, 1),
        "forecast_rainfall_mm": forecast_rain_mm,
        "current_moisture_pct": theta_s,
        "recommended_irrigation_depth_mm": irrigation_vol_mm,
        "advisory_status": status
    }

# -------------------------------------------------------------
# MODULE 4: Multi-Channel Market Strategy (Live Mandi API Bridge)
# -------------------------------------------------------------
MARKET_DATABASE = {
    "Rice": {
        "current_year_rate": 2650,
        "prev_year_rate": 2380,
        "msp": 2300,
        "7d_forecast": 2780,
        "trend": "Bullish (+11.3% YoY)",
        "channels": [
            {"channel": "Govt APMC Mandi", "buyer": "Licensed Wholesale Arhtiyas", "rate_per_qtl": 2650, "deductions_pct": 2.5, "speed": "1-2 days (Cash/NEFT)", "location": "Koyambedu APMC / Kanchipuram"},
            {"channel": "e-NAM National Portal", "buyer": "Interstate Millers & Corporates", "rate_per_qtl": 2790, "deductions_pct": 1.0, "speed": "3-4 days (Direct Escrow)", "location": "Online National Market"},
            {"channel": "Direct Consumer / Uzhavar Sandhai", "buyer": "Retail Consumers & Local Hoteliers", "rate_per_qtl": 3100, "deductions_pct": 0.0, "speed": "Immediate Daily Cash", "location": "Local Farmer Market (T. Nagar/Ambattur)"}
        ],
        "action": "HOLD or sell via Direct Farmer Markets for highest gross margin."
    },
    "Tomato": {
        "current_year_rate": 2200,
        "prev_year_rate": 2900,
        "msp": 1200,
        "7d_forecast": 1850,
        "trend": "Bearish (-24.1% YoY)",
        "channels": [
            {"channel": "Govt APMC Mandi", "buyer": "Bulk Vegetable Commission Agents", "rate_per_qtl": 2200, "deductions_pct": 5.0, "speed": "Immediate auction", "location": "Koyambedu Wholesale Hub"},
            {"channel": "Direct Consumer / Uzhavar Sandhai", "buyer": "Households & Local Canteens", "rate_per_qtl": 2800, "deductions_pct": 0.0, "speed": "Immediate Daily Cash", "location": "District Uzhavar Sandhai"},
            {"channel": "Food Processing Co.", "buyer": "Sauce & Puree Manufacturers", "rate_per_qtl": 2350, "deductions_pct": 1.5, "speed": "7-day invoice", "location": "Industrial Agri-Cluster"}
        ],
        "action": "SELL IMMEDIATELY; supply arrivals increasing rapidly, price dropping."
    },
    "Maize": {
        "current_year_rate": 2620,
        "prev_year_rate": 2240,
        "msp": 2090,
        "7d_forecast": 2650,
        "trend": "Bullish (+16.9% YoY)",
        "channels": [
            {"channel": "Govt APMC Mandi", "buyer": "Feed Mill Procurement Agents", "rate_per_qtl": 2620, "deductions_pct": 2.0, "speed": "2 days", "location": "Kallakurichi / Madurai RM"},
            {"channel": "Poultry Farm Tie-up", "buyer": "Direct Poultry Feed Integrators", "rate_per_qtl": 2800, "deductions_pct": 0.5, "speed": "Weekly bank transfer", "location": "Namakkal / Erode Belt"},
            {"channel": "e-NAM Portal", "buyer": "Starch & Ethanol Processors", "rate_per_qtl": 2710, "deductions_pct": 1.0, "speed": "3 days", "location": "National Digital Exchange"}
        ],
        "action": "HOLD for Poultry Integrator contracts or list on e-NAM."
    },
    "Cotton": {
        "current_year_rate": 7450,
        "prev_year_rate": 7100,
        "msp": 7121,
        "7d_forecast": 7520,
        "trend": "Stable (+4.9% YoY)",
        "channels": [
            {"channel": "Cotton Corporation of India (CCI)", "buyer": "Government Procurement Centre", "rate_per_qtl": 7121, "deductions_pct": 0.0, "speed": "Direct Bank Deposit", "location": "CCI Center (Villupuram)"},
            {"channel": "Textile Spinning Mills", "buyer": "Coimbatore / Tiruppur Ginning Mills", "rate_per_qtl": 7600, "deductions_pct": 1.5, "speed": "3-5 days", "location": "Coimbatore Textile Corridor"},
            {"channel": "Govt APMC Mandi", "buyer": "Registered Cotton Commission Agents", "rate_per_qtl": 7450, "deductions_pct": 2.0, "speed": "1-2 days", "location": "Theni / Perambalur Mandi"}
        ],
        "action": "SELL to Direct Textile Spinning Mills to surpass standard APMC rates."
    },
    "Potato": {
        "current_year_rate": 1650,
        "prev_year_rate": 1420,
        "msp": 1150,
        "7d_forecast": 1720,
        "trend": "Bullish (+16.2% YoY)",
        "channels": [
            {"channel": "Govt APMC Mandi", "buyer": "Root Crop Wholesalers", "rate_per_qtl": 1650, "deductions_pct": 3.0, "speed": "Same day cash", "location": "Salem / Mettupalayam APMC"},
            {"channel": "Cold Storage Depot", "buyer": "Snack Food Processors", "rate_per_qtl": 1850, "deductions_pct": 1.0, "speed": "Contract billing", "location": "Nilgiris / Ooty Supply Center"},
            {"channel": "Direct Consumer / Uzhavar Sandhai", "buyer": "Retail Consumers", "rate_per_qtl": 2100, "deductions_pct": 0.0, "speed": "Daily cash", "location": "Local Uzhavar Sandhai"}
        ],
        "action": "HOLD; cold storage buffer viable, prices forecasted to climb."
    },
    "Chickpea": {
        "current_year_rate": 5850,
        "prev_year_rate": 5300,
        "msp": 5440,
        "7d_forecast": 6050,
        "trend": "Bullish (+10.4% YoY)",
        "channels": [
            {"channel": "Govt APMC Mandi", "buyer": "Pulse Traders Association", "rate_per_qtl": 5850, "deductions_pct": 2.0, "speed": "1-2 days", "location": "Dindigul APMC"},
            {"channel": "Dal Mill Processors", "buyer": "Commercial Pulse Mills", "rate_per_qtl": 6150, "deductions_pct": 1.0, "speed": "3 days", "location": "Madurai / Virudhunagar Hub"},
            {"channel": "e-NAM Portal", "buyer": "Institutional Bulk Procurement", "rate_per_qtl": 5980, "deductions_pct": 0.8, "speed": "Escrow transfer", "location": "National Digital Market"}
        ],
        "action": "HOLD for festive peak demand or sell directly to Dal Mills."
    }
}

class EnhancedMarketRequest(BaseModel):
    crop_name: str = Field(default="Rice", example="Rice")
    state: str = Field(default="Andhra Pradesh", min_length=2, max_length=80, example="Andhra Pradesh")
    district: str | None = Field(default=None, max_length=80, example="Nellore")
    harvest_yield_quintals: float = Field(default=30.0, ge=1.0, le=1000.0, example=30.0)
    distance_to_mandi_km: float = Field(default=15.0, ge=0.0, le=250.0, example=15.0)

async def get_live_market_modal_price(crop_name: str, state: str, fallback_price: float, district: str | None = None):
    records, error = await fetch_agmarknet_state_records(state, district)
    matches = matching_market_records(records, crop_name)
    if not matches:
        return fallback_price, False, None, error or "no_data"
    latest = max(matches, key=lambda row: parse_market_date(row.get("arrival_date")))
    return market_price_number(latest.get("modal_price")) or fallback_price, True, latest, None

@app.post("/api/v1/market-strategy")
async def evaluate_market_strategy(req: EnhancedMarketRequest):
    crop_key = find_supported_live_market_crop(req.crop_name) or req.crop_name.title()
    if crop_key not in MARKET_DATABASE:
        raise HTTPException(status_code=400, detail="Choose one of the supported crops: Rice, Tomato, Maize, Cotton, Potato, or Chickpea.")
    data = MARKET_DATABASE[crop_key]

    curr_rate, has_live_market_rate, live_record, live_feed_status = await get_live_market_modal_price(
        crop_key, req.state, data["current_year_rate"], req.district
    )
    prev_rate = data["prev_year_rate"]
    msp = data["msp"]
    proj_rate = data["7d_forecast"]

    yoy_diff = curr_rate - prev_rate
    yoy_pct = round((yoy_diff / prev_rate) * 100, 2)

    gross_current = round(req.harvest_yield_quintals * curr_rate, 2)
    gross_projected = round(req.harvest_yield_quintals * proj_rate, 2)

    logistics_cost = round((req.distance_to_mandi_km / 10.0) * (8.0 * req.harvest_yield_quintals), 2)
    net_delay_gain = round(gross_projected - gross_current - logistics_cost, 2)

    decision = "HOLD (7-10 DAYS)" if net_delay_gain > 0 and "Bullish" in data["trend"] else "SELL NOW"

    analyzed_channels = []
    for ch in data["channels"]:
        rate = curr_rate if has_live_market_rate and ch["channel"] == "Govt APMC Mandi" else ch["rate_per_qtl"]
        market_location = ch["location"]
        if has_live_market_rate and live_record and ch["channel"] == "Govt APMC Mandi":
            market_location = " / ".join(part for part in (live_record.get("market"), live_record.get("district"), req.state) if part)
        gross = round(rate * req.harvest_yield_quintals, 2)
        comm_fee = round(gross * (ch["deductions_pct"] / 100.0), 2)
        net_proceeds = round(gross - comm_fee - logistics_cost, 2)
        analyzed_channels.append({
            "channel_type": ch["channel"],
            "target_buyer": ch["buyer"],
            "market_location": market_location,
            "offered_rate_inr_qtl": rate,
            "deductions_pct": ch["deductions_pct"],
            "payment_speed": ch["speed"],
            "estimated_net_payout_inr": net_proceeds
        })

    analyzed_channels.sort(key=lambda x: x["estimated_net_payout_inr"], reverse=True)
    best_channel = analyzed_channels[0]

    db = get_db()
    if db:
        try:
            record = MarketRecord(
                crop_name=crop_key,
                harvest_yield_quintals=req.harvest_yield_quintals,
                distance_to_mandi_km=req.distance_to_mandi_km,
                current_year_rate=curr_rate,
                previous_year_rate=prev_rate,
                optimal_decision=decision,
                best_channel=best_channel["channel_type"]
            )
            db.add(record)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"SQL Insert Error (MarketRecord): {e}")
        finally:
            db.close()

    return {
        "crop": crop_key,
        "state": req.state,
        "optimal_decision": decision,
        "best_selling_channel": best_channel["channel_type"],
        "recommended_action": data["action"],
        "price_analytics": {
            "current_year_modal_2026": curr_rate,
            "current_price_source": "AGMARKNET via data.gov.in" if has_live_market_rate else "Bundled benchmark (not live)",
            "current_price_is_live": has_live_market_rate,
            "current_price_report_date": live_record.get("arrival_date") if live_record else None,
            "current_price_market": live_record.get("market") if live_record else None,
            "current_price_district": live_record.get("district") if live_record else None,
            "current_price_min": market_price_number(live_record.get("min_price")) if live_record else None,
            "current_price_max": market_price_number(live_record.get("max_price")) if live_record else None,
            "live_feed_status": live_feed_status,
            "previous_year_modal_2025": prev_rate,
            "previous_price_source": "Bundled planning benchmark; not historical mandi data",
            "yoy_price_change_inr": yoy_diff,
            "yoy_price_change_pct": yoy_pct,
            "government_msp": msp,
            "projected_7d_rate": proj_rate,
            "market_momentum": data["trend"]
        },
        "logistics_and_harvest": {
            "volume_quintals": req.harvest_yield_quintals,
            "distance_km": req.distance_to_mandi_km,
            "estimated_freight_cost_inr": logistics_cost
        },
        "selling_channels_comparison": analyzed_channels
    }

# -------------------------------------------------------------
# MODULE 5: Dynamic Multilingual Chatbot & Voice Assistant
# -------------------------------------------------------------
class ChatHistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class ChatbotRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, example="వరి పంట మార్కెట్ రేటు ఎంత?")
    language: str = Field(default="auto", example="te-IN")
    history: list[ChatHistoryTurn] = Field(default_factory=list, max_length=10)


class SpeechRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=6000)
    language: str = Field(default="te-IN", max_length=10)

MULTILINGUAL_KNOWLEDGE = [
    {
        "keywords": ["hi", "hello", "hey", "namaste", "నమస్కారం", "బాగున్నారా", "హలో", "नमस्ते", "வணக்கம்", "வணக்கம்"],
        "en": "Hello! I am your Smart AgriGuide Agricultural Intelligence Assistant. You can ask me about live mandi crop prices, fertilizer recommendations, disease identification for lemon, tomato, and rice, or smart irrigation water schedules. How can I assist your farm today?",
        "te": "నమస్కారం! నేను మీ స్మార్ట్ అగ్రిగైడ్ వ్యవసాయ సహాయకుడిని. మీరు నన్ను నేటి మార్కెట్ ధరలు, ఎరువుల మోతాదు (N-P-K), టమాట మరియు నిమ్మ ఆకు తెగుళ్ల నివారణ మందులు, లేదా వర్ష సూచనతో కూడిన నీటిపారుదల సలహాల గురించి అడగవచ్చు. మీ వ్యవసాయానికి నేను ఎలా సహాయపడగలను?",
        "hi": "नमस्ते! मैं आपका स्मार्ट एग्रीगाइड कृषि सहायक हूँ। आप मुझसे आज के मंडी भाव, एन-पी-के उर्वरक की सही मात्रा, टमाटर और नींबू के रोगों के उपचार या मौसम आधारित सिंचाई के बारे में पूछ सकते हैं। आज मैं आपकी क्या मदद कर सकता हूँ?",
        "ta": "வணக்கம்! நான் உங்கள் ஸ்மார்ட் அக்ரிகைட் விவசாய உதவியாளர். இன்றைய சந்தை விலை நிலவரம், உர பரிந்துரை, தக்காளி மற்றும் எலுமிச்சை இலை நோய் மேலாண்மை, அல்லது நீர்ப்பாசன ஆலோசனை பற்றி நீங்கள் கேட்கலாம். இன்று நான் உங்களுக்கு எவ்வாறு உதவட்டும்?"
    },
    {
        "keywords": ["rice", "paddy", "వరి", "ధాన్యం", "బియ్యం", "అరిసి", "vari", "vadlu", "dhanyam", "biyyam", "அரிசி", "धान", "चावल"],
        "en": "For Rice (Paddy), current 2026 mandi modal rates are ₹2,650 per quintal compared to ₹2,380 per quintal in 2025 (+11.3% YoY). The 7-day outlook is bullish at ₹2,780 per quintal. If you have farm storage, hold for 7-10 days, or sell via Direct Farmer Markets (Uzhavar Sandhai / Rythu Bazaar) for maximum net profit without broker commission.",
        "te": "వరి (ధాన్యం) పంటకి సంబంధించి, 2026 నాటి ప్రస్తుత మార్కెట్ ధర క్వింటాలుకు ₹2,650 గా ఉంది (2025 తో పోలిస్తే +11.3% పెరిగింది). రాబోయే 7 రోజుల్లో ధర క్వింటాలుకు ₹2,780 వరకు పెరిగే అవకాశం ఉంది. మీ వద్ద నిల్వ సదుపాయం ఉంటే 7-10 రోజులు ఆగి అమ్మడం లేదా రైతు బజార్ ద్వారా నేరుగా అమ్మడం వల్ల దళారీల కమీషన్ లేకుండా పూర్తి లాభం పొందవచ్చు.",
        "hi": "धान (चावल) के लिए 2026 का वर्तमान मंडी भाव ₹2,650 प्रति क्विंटल है (2025 की तुलना में +11.3% अधिक वृद्धि)। अगले 7 दिनों में भाव ₹2,780 तक बढ़ने का अनुमान है। यदि आपके पास भंडारण है, तो 7-10 दिन रुकें या सीधे किसान बाजार (रैथू बाज़ार) में बेचें ताकि दलालों की कटौती से बच सकें।",
        "ta": "நெல் பயிருக்கு 2026 ஆம் ஆண்டு சந்தை சராசரி விலை குவிண்டாலுக்கு ₹2,650 ஆக உள்ளது (+11.3% உயர்வு). அடுத்த 7 நாட்களில் விலை ₹2,780 வரை உயர வாய்ப்புள்ளது. உங்களிடம் சேமிப்பு வசதி இருந்தால் 7-10 நாட்கள் காத்திருக்கலாம் அல்லது இடைத்தரகர் கமிஷன் இல்லாமல் உழவர் சந்தை மூலம் நேரடியாக விற்பனை செய்து அதிக லாபம் பெறலாம்."
    },
    {
        "keywords": ["lemon", "citrus", "నిమ్మ", "ఎలుమిచ్చై", "nimma", "miner", "canker", "chetu", "నీंबू"],
        "en": "For Lemon and Citrus trees, the two primary foliar issues are Citrus Leaf Miner (curled leaves with silvery serpentine trails) and Citrus Canker (raised corky brown spots with yellow halos). For Leaf Miner, spray Neem oil (5ml/L) or Spinosad (0.3ml/L) during new vegetative leaf flush. For bacterial canker, apply Copper Oxychloride (3g/L) combined with Streptocycline (1g/10L) and prune diseased twigs.",
        "te": "నిమ్మ చెట్లకు ప్రధానంగా రెండు సమస్యలు వస్తాయి: 1) ఆకుముడత మరియు ఆకుతొలుచు పురుగు (Leaf Miner) - దీనివల్ల ఆకులు ముడుచుకుని తెల్లటి వెండి గీతలు ఏర్పడతాయి. నివారణకు కొత్త చిగురు వచ్చే సమయంలో వేపనూనె (5 మి.లీ/లీ) లేదా స్పినోసాడ్ పిచికారీ చేయండి. 2) నిమ్మ క్యాంకర్ తెగులు - ఆకులపై పసుపు రంగు అంచులతో గోధుమ రంగు మచ్చలు వస్తే కాపర్ ఆక్సిక్లోరైడ్ (3 గ్రా/లీ) కలిపి పిచికారీ చేయండి.",
        "hi": "नींबू के पौधों में मुख्य रूप से दो रोग होते हैं: 1) पत्ती मोड़क कीट (Leaf Miner) - इसमें पत्तियां मुड़ जाती हैं और सफेद टेढ़ी-मेढ़ी लकीरें बन जाती हैं। रोकथाम के लिए नीम तेल (5ml/L) या स्पिनोसैड का छिड़काव करें। 2) नींबू का कैंकर रोग - पत्तियों पर पीले छल्ले वाले भूरे धब्बे होने पर कॉपर ऑक्सीक्लोराइड (3g/L) का छिड़काव करें और सूखी टहनियों की छंटाई करें।",
        "ta": "எலுமிச்சை மரங்களில் இரண்டு முக்கிய பிரச்சனைகள் ஏற்படுகின்றன: 1) இலை சுரங்கப் பூச்சி - இலைகள் சுருங்கி வெள்ளி நிற வளைந்த கோடுகள் தோன்றும். புதிய தளிர்கள் வரும்போது வேப்ப எண்ணெய் (5 மி.லி/லி) அல்லது ஸ்பினோசாட் தெளிக்கவும். 2) பாக்டீரியா புள்ளி நோய் (Citrus Canker) - மஞ்சள் வளையத்துடன் கூடிய பழுப்பு நிற புள்ளிகள் இருந்தால் காப்பர் ஆக்ஸிகுளோரைடு (3 கிராம்/லி) தெளிக்கவும்."
    },
    {
        "keywords": ["tomato", "టమాట", "తమటా", "தக்காளி", "tamata", "thota", "टमाटर"],
        "en": "Tomato wholesale prices are currently volatile at ₹2,200 per quintal due to heavy peak arrivals (-24.1% YoY). To prevent spoilage and financial loss, it is strongly advised to SELL IMMEDIATELY at local wholesale mandis or supply food processing puree factories. For leaf blights, spray Mancozeb (2.5g/L) every 8-10 days.",
        "te": "టమాట పంట మార్కెట్ ధర ప్రస్తుతం మార్కెట్‌కు సరుకు అధికంగా రావడం వల్ల తగ్గింది (క్వింటాలుకు ₹2,200, గత ఏడాదితో పోలిస్తే -24.1%). పంట పాడవకుండా ఉండటానికి మరియు నష్టాలు నివారించడానికి వెంటనే స్థానిక మార్కెట్లో లేదా ఫుడ్ ప్రాసెసింగ్ యూనిట్లలో అమ్మడం ఉత్తమం. ఆకుమచ్చ తెగులు నివారణకు మాంకోజెబ్ (2.5 గ్రా/లీ) పిచికారీ చేయండి.",
        "hi": "टमाटर की भारी आवक के कारण थोक भाव में गिरावट आई है (-24.1% YoY, ₹2,200 प्रति क्विंटल)। फसल को खराब होने से बचाने के लिए इसे तुरंत स्थानीय मंडी या सॉस/प्यूरी बनाने वाली कंपनियों को बेचें। पत्तियों पर धब्बों (झुलसा रोग) से बचाव के लिए मैंकोजेब (2.5g/L) का छिड़काव करें।",
        "ta": "தக்காளி வரத்து அதிகமாக இருப்பதால் மொத்த சந்தை விலை குறைந்துள்ளது (குவிண்டாலுக்கு ₹2,200). நஷ்டத்தை தவிர்க்க உடனடியாக மொத்த சந்தையிலோ அல்லது பதப்படுத்தும் நிறுவனங்களிலோ விற்பனை செய்ய பரிந்துரைக்கப்படுகிறது. இலை கருகல் நோய்க்கு மான்கோசெப் (2.5 கிராம்/லி) தெளிக்கவும்."
    },
    {
        "keywords": ["fertilizer", "npk", "urea", "dap", "potash", "ఎరువులు", "యూరియా", "పోటాష్", "eruvulu", "mandhu", "उर्वरक", "खाद", "உரம்"],
        "en": "Balanced soil fertilization depends on crop requirements: Rice generally requires N-P-K in an 80:40:40 kg/ha ratio, while Cotton requires 120:60:60 kg/ha. Apply full Phosphorus (DAP/SSP) and Potash (MOP) as basal dose at planting, and split Nitrogen (Urea) into 2-3 top-dressings at active tillering and flowering stages.",
        "te": "సమతుల్య ఎరువుల యాజమాన్యం కోసం: వరి పంటకు సాధారణంగా హెక్టారుకు 80:40:40 కిలోల N-P-K అవసరం, పత్తికి 120:60:60 కిలోలు అవసరం. భాస్వరం (DAP/SSP) మరియు పొటాష్ (MOP) లను ఆఖరి దుక్కిలో వేయాలి. నత్రజని (యూరియా) ని సమాన భాగాలుగా విభజించి నాటిన 20-25 రోజులకు మరియు పిలకలు వచ్చే దశలో వేయాలి.",
        "hi": "संतुलित पोषण के लिए: धान की फसल में एन-पी-के (N-P-K) 80:40:40 किग्रा/हेक्टेयर तथा कपास में 120:60:60 का अनुपात रखें। डीएपी (DAP) और पोटाश (MOP) की पूरी मात्रा बुवाई के समय बेसल डोज के रूप में दें, जबकि यूरिया (नाइट्रोजन) को 2-3 बार में कल्ले फूटते समय और फूल आते समय दें।",
        "ta": "சமச்சீர் உர நிர்வாகத்திற்கு: நெல் பயிருக்கு ஒரு ஹெக்டேருக்கு 80:40:40 கிலோ N-P-K தேவைப்படுகிறது, பருத்திக்கு 120:60:60 கிலோ தேவை. பாஸ்பரஸ் (DAP) மற்றும் பொட்டாஷ் முழு அளவையும் அடியுரமாக இடவும். தழைச்சத்தை (யுரியா) 2-3 தவணைகளாக பிரித்து தூர்கட்டும் பருவம் மற்றும் பூக்கும் பருவத்தில் இடவும்."
    },
    {
        "keywords": ["blight", "disease", "fungus", "రోగం", "తెగులు", "మచ్చలు", "tegulu", "machalu", "நோய்", "रोग", "बीमारी"],
        "en": "For leaf blights and fungal infections: Early Blight produces concentric brown ring spots (control with Mancozeb 75 WP at 2g/L or Chlorothalonil). Late Blight causes rapid dark water-soaked rot (spray systemic Metalaxyl-M + Mancozeb at 2.5g/L). Always prune lower affected foliage and avoid overhead sprinkler watering during disease outbreaks.",
        "te": "ఆకు తెగుళ్లు మరియు శిలీంధ్ర వ్యాధులకు: ఎర్లీ బ్లైట్ తెగులు ఆకులపై వలయాల వంటి గోధుమ మచ్చలను కలిగిస్తుంది (నివారణకు మాంకోజెబ్ 2 గ్రా/లీ లేదా క్లోరోథలోనిల్ పిచికారీ చేయండి). లేట్ బ్లైట్ ఆకులపై నల్లటి తడి మచ్చలతో వేగంగా వ్యాపిస్తుంది (నివారణకు మెటలాక్సిల్-M 2.5 గ్రా/లీ పిచికారీ చేయండి). రోగగ్రస్త ఆకులను తొలగించి, పైనుంచి నీరు చల్లే పద్ధతిని నివారించండి.",
        "hi": "पत्तियों के झुलसा रोग के लिए: अगेती झुलसा में पत्तियों पर गोल छल्लेदार भूरे धब्बे बनते हैं (इसके लिए मैंकोजेब 2g/L का छिड़काव करें)। पछेती झुलसा में पत्तियां तेजी से काली पड़कर सड़ने लगती हैं (इसके लिए मेटालेक्सिल-M 2.5g/L का छिड़काव करें)। रोगग्रस्त पत्तियों को तोड़कर नष्ट कर दें और शाम के समय पानी देने से बचें।",
        "ta": "இலை கருகல் நோய்களுக்கு: ஆரம்ப கருகல் நோய்க்கு இலைகளில் வளைய வடிவ புள்ளிகள் தோன்றும் (மான்கோசெப் 2 கிராம்/லி தெளிக்கவும்). பிந்தைய கருகல் நோய் இலைகளை கருமையாக்கி அழுகச் செய்யும் (மெட்டலாக்சில்-M 2.5 கிராம்/லி தெளிக்கவும்). பாதிக்கப்பட்ட இலைகளை அகற்றி அழிக்கவும்."
    },
    {
        "keywords": ["irrigation", "water", "rain", "weather", "నీరు", "వర్షం", "నీటిపారుదల", "neellu", "varsham", "தண்ணீர்", "பாசனம்", "पानी", "सिंचाई"],
        "en": "Smart AgriGuide continuously integrates soil moisture sensors with 24-hour Open-Meteo precipitation forecasts. If local rainfall probability exceeds 65%, automated irrigation is preemptively suppressed to conserve groundwater and electricity. During dry spells, deliver water to restore field capacity based on crop root zone depth.",
        "te": "స్మార్ట్ అగ్రిగైడ్ నేలలోని తేమను మరియు రాబోయే 24 గంటల వర్ష సూచనను స్వయంచాలకంగా విశ్లేషిస్తుంది. వర్షం పడే అవకాశం 65% కంటే ఎక్కువ ఉన్నప్పుడు, మోటార్ పంపును ఆపివేసి నీరు మరియు విద్యుత్‌ను ఆదా చేస్తుంది. వర్షం లేనప్పుడు నేల తేమ ఫీల్డ్ కెపాసిటీ కంటే తగ్గకుండా తగినంత నీటిని అందించాలని సూచిస్తుంది.",
        "hi": "स्मार्ट एग्रीगाइड मिट्टी की नमी और मौसम के 24 घंटे के बारिश पूर्वानुमान का विश्लेषण करता है। यदि बारिश की संभावना 65% से अधिक होती है, तो यह अनावश्यक सिंचाई को रोक देता है जिससे पानी और बिजली की बचत होती है। सूखे मौसम में फसल की जड़ की गहराई के अनुसार पर्याप्त पानी देने की सलाह दी जाती है।",
        "ta": "ஸ்மார்ட் அக்ரிகைட் மண்ணின் ஈரப்பதத்தையும் 24 மணி நேர மழை முன்னறிவிப்பையும் கண்காணிக்கிறது. மழை பெய்ய 65%க்கு மேல் வாய்ப்பு இருந்தால் மின்சாரம் மற்றும் தண்ணீரை சேமிக்க நீர்ப்பாசனத்தை நிறுத்தி வைக்கிறது. வறண்ட காலங்களில் பயிரின் வேர் ஆழத்திற்கு ஏற்ப தகுந்த நீர்ப்பாசனம் செய்ய பரிந்துரைக்கிறது."
    },
    {
        "keywords": ["cotton", "పత్తి", "పత్తిపంట", "பருத்தி", "patti", "kapas", "कपास"],
        "en": "Cotton modal price currently trades around ₹7,450 per quintal against the government MSP floor of ₹7,121. Selling directly to textile spinning mills in industrial clusters (Coimbatore, Tiruppur, or Guntur) yields up to ₹7,600 per quintal with minimal middleman commission deductions.",
        "te": "పత్తి ప్రస్తుత మార్కెట్ సగటు ధర క్వింటాలుకు ₹7,450 గా ఉంది (ప్రభుత్వ మద్దతు ధర ₹7,121). దళారులకు కాకుండా గుంటూరు లేదా కోయంబత్తూరు వంటి టెక్స్‌టైల్ జిన్నింగ్ మిల్లులకు నేరుగా విక్రయించడం ద్వారా క్వింటాలుకు ₹7,600 వరకు నికర ఆదాయం పొందవచ్చు.",
        "hi": "कपास का वर्तमान बाजार भाव लगभग ₹7,450 प्रति क्विंटल है (सरकार का न्यूनतम समर्थन मूल्य ₹7,121 है)। बिचौलियों के बजाय सीधे टेक्सटाइल कताई मिलों को बेचने से ₹7,600 प्रति क्विंटल तक का शुद्ध लाभ प्राप्त किया जा सकता है।",
        "ta": "பருத்தி சராசரி விலை தற்போது குவிண்டாலுக்கு ₹7,450 ஆக உள்ளது (அரசு ஆதார விலை ₹7,121). இடைத்தரகர்களை தவிர்த்து கோவை, திருப்பூர் நூற்பாலைகளுக்கு நேரடியாக விற்பனை செய்வதன் மூலம் குவிண்டாலுக்கு ₹7,600 வரை நிகர வருமானம் பெறலாம்."
    }
]

def detect_language(text: str, user_lang_code: str) -> str:
    # 1. Prioritize explicit user language selection from the dropdown
    if user_lang_code:
        clean_code = user_lang_code.lower()
        if "te" in clean_code:
            return "te"
        elif "hi" in clean_code:
            return "hi"
        elif "ta" in clean_code:
            return "ta"
        elif "en" in clean_code:
            return "en"

    # 2. Check Unicode character ranges
    for ch in text:
        code = ord(ch)
        if 0x0C00 <= code <= 0x0C7F:
            return "te"
        elif 0x0900 <= code <= 0x097F:
            return "hi"
        elif 0x0B80 <= code <= 0x0BFF:
            return "ta"

    # 3. Transliterated / Romanized phonetic Telugu keywords
    telugu_roman_words = [
        "vari", "vadlu", "dhanyam", "biyyam", "panta", "dharalu", "entha", 
        "ammala", "vadda", "mandhu", "eruvulu", "neellu", "varsham", 
        "nimma", "chettu", "tamata", "thota", "kura", "aaku", "tegulu", "machalu", "patti"
    ]
    lower_text = text.lower()
    if any(w in lower_text for w in telugu_roman_words):
        return "te"

    return "en"

def search_local_mandi_prices(commodity: str, district: str | None, state: str | None = None) -> dict:
    csv_path = os.path.join(os.path.dirname(__file__), "Crop_recommendation1.csv")
    if not os.path.exists(csv_path):
        return {"error": "The local mandi-price snapshot is unavailable."}

    query = commodity.casefold().strip()
    if not query:
        return {"error": "Provide a crop or commodity name."}
    district_query = district.casefold().strip() if district else ""
    aliases = {
        "rice": ["rice", "paddy"],
        "paddy": ["paddy", "rice"],
        "maize": ["maize", "corn"],
        "corn": ["corn", "maize"],
        "groundnut": ["groundnut", "peanut"],
        "peanut": ["peanut", "groundnut"],
    }
    terms = aliases.get(query, [query])
    records = []
    state_query = state.casefold().strip() if state else ""
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                row_commodity = (row.get("Commodity") or "").casefold().strip()
                row_district = (row.get("District") or "").casefold().strip()
                row_state = (row.get("State") or "").casefold().strip()
                if not any(term in row_commodity or row_commodity in term for term in terms):
                    continue
                if state_query and state_query not in row_state:
                    continue
                if district_query and district_query not in row_district:
                    continue

                raw_date = (row.get("Arrival_Date") or "").strip()
                try:
                    sortable_date = datetime.strptime(raw_date, "%d/%m/%Y").date().isoformat()
                except ValueError:
                    sortable_date = raw_date
                records.append({
                    "commodity": row.get("Commodity"),
                    "state": row.get("State"),
                    "district": row.get("District"),
                    "market": row.get("Market"),
                    "arrival_date": raw_date,
                    "min_price_inr_per_quintal": row.get("Min_x0020_Price"),
                    "max_price_inr_per_quintal": row.get("Max_x0020_Price"),
                    "modal_price_inr_per_quintal": row.get("Modal_x0020_Price"),
                    "_sort_date": sortable_date,
                })
    except OSError:
        return {"error": "The local mandi-price snapshot could not be read."}

    records.sort(key=lambda item: item["_sort_date"], reverse=True)
    for record in records:
        record.pop("_sort_date", None)
    return {
        "source": "Crop_recommendation1.csv local snapshot",
        "state": state,
        "live_data_available": False,
        "records": records[:5],
        "message": "No matching mandi records were found." if not records else None,
    }


def find_supported_live_market_crop(commodity: str) -> str | None:
    query = "".join(character for character in commodity.casefold() if character.isalnum())
    for crop_name, aliases in MARKET_COMMODITY_ALIASES.items():
        candidates = {"".join(character for character in value.casefold() if character.isalnum()) for value in aliases}
        candidates.add("".join(character for character in crop_name.casefold() if character.isalnum()))
        if any(query == candidate or query.startswith(candidate) for candidate in candidates):
            return crop_name
    return None


async def get_agent_mandi_prices(commodity: str, state: str, district: str | None = None) -> dict:
    crop_name = find_supported_live_market_crop(commodity)
    feed_status = None
    if crop_name:
        records, feed_status = await fetch_agmarknet_state_records(state, district)
        live_matches = matching_market_records(records, crop_name)
        if district:
            district_query = district.casefold().strip()
            live_matches = [
                record for record in live_matches
                if district_query in str(record.get("district", "")).casefold()
            ]
        live_matches.sort(key=lambda row: parse_market_date(row.get("arrival_date")), reverse=True)
        if live_matches:
            return {
                "source": "AGMARKNET via data.gov.in",
                "state": state,
                "live_data_available": True,
                "records": [{
                    "commodity": record.get("commodity"),
                    "state": record.get("state", state),
                    "district": record.get("district"),
                    "market": record.get("market"),
                    "arrival_date": record.get("arrival_date"),
                    "min_price_inr_per_quintal": market_price_number(record.get("min_price")),
                    "max_price_inr_per_quintal": market_price_number(record.get("max_price")),
                    "modal_price_inr_per_quintal": market_price_number(record.get("modal_price")),
                } for record in live_matches[:5]],
                "message": None,
            }

    snapshot = await asyncio.to_thread(search_local_mandi_prices, commodity, district, state)
    snapshot["live_feed_status"] = feed_status or ("no_matching_live_records" if crop_name else "commodity_not_supported_by_live_feed")
    if snapshot.get("records"):
        snapshot["message"] = "An official daily quote is not available for this request. These results are a local CSV snapshot, not live market prices."
    elif crop_name:
        snapshot["message"] = "No matching official daily quote or local snapshot was found for this crop and location."
    return snapshot


def _recommend_crop_for_agent(arguments: dict) -> dict:
    if crop_model is None:
        return {"error": "The crop recommendation model is not loaded."}
    required_values = ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall"]
    missing = [key for key in required_values if arguments.get(key) is None]
    if missing:
        readable = {
            "nitrogen": "soil nitrogen (N)",
            "phosphorus": "soil phosphorus (P)",
            "potassium": "soil potassium (K)",
            "temperature": "temperature in Celsius",
            "humidity": "relative humidity percentage",
            "ph": "soil pH",
            "rainfall": "rainfall in millimeters",
        }
        return {"needs_more_info": [readable[key] for key in missing]}
    data = CropFeatures(**arguments)
    features = np.array([[
        data.nitrogen, data.phosphorus, data.potassium,
        data.temperature, data.humidity, data.ph, data.rainfall
    ]])
    probabilities = crop_model.predict_proba(features)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]
    return {
        "primary_recommendation": str(crop_model.predict(features)[0]).title(),
        "top_recommendations": [
            {"crop": str(crop_model.classes_[index]).title(), "confidence": round(float(probabilities[index]) * 100, 2)}
            for index in top_indices
        ],
        "input_summary": data.model_dump(),
        "source": "Smart AgriGuide crop recommendation model",
    }


async def execute_agri_agent_tool(name: str, arguments: dict) -> dict:
    if name == "recommend_crop":
        return await asyncio.to_thread(_recommend_crop_for_agent, arguments)

    if name == "get_weather":
        latitude = arguments.get("latitude")
        longitude = arguments.get("longitude")
        using_default_location = latitude is None or longitude is None
        latitude = 13.0827 if latitude is None else float(latitude)
        longitude = 80.2707 if longitude is None else float(longitude)
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            return {"error": "Latitude must be between -90 and 90 and longitude between -180 and 180."}
        result = await get_live_weather_telemetry(latitude, longitude)
        result["location_note"] = "Project default coordinates near Chennai were used." if using_default_location else "User-provided coordinates were used."
        return result

    if name == "get_irrigation_advice":
        moisture = arguments.get("soil_moisture_pct")
        if moisture is None:
            return {"needs_more_info": ["current soil moisture percentage"], "message": "Ask the farmer for the current soil-moisture reading before calculating irrigation."}
        field_capacity = arguments.get("field_capacity_pct")
        root_depth = arguments.get("root_zone_depth_mm")
        latitude = arguments.get("latitude")
        longitude = arguments.get("longitude")
        assumptions = []
        if field_capacity is None:
            field_capacity = 35.0
            assumptions.append("default field capacity: 35%")
        if root_depth is None:
            root_depth = 300.0
            assumptions.append("default root-zone depth: 300 mm")
        if latitude is None or longitude is None:
            latitude = 13.0827 if latitude is None else float(latitude)
            longitude = 80.2707 if longitude is None else float(longitude)
            assumptions.append("project default coordinates near Chennai")
        if not -90 <= float(latitude) <= 90 or not -180 <= float(longitude) <= 180:
            return {"error": "Latitude must be between -90 and 90 and longitude between -180 and 180."}
        telemetry = IrrigationTelemetry(
            soil_moisture_pct=float(moisture),
            field_capacity_pct=float(field_capacity),
            root_zone_depth_mm=float(root_depth),
            latitude=float(latitude),
            longitude=float(longitude),
        )
        result = await calculate_irrigation(telemetry)
        result["assumptions"] = assumptions
        return result

    if name == "get_mandi_prices":
        state = arguments.get("state")
        if not isinstance(state, str) or not state.strip():
            return {"needs_more_info": ["state or union territory"], "message": "Ask which state or union territory to search before looking up mandi prices."}
        return await get_agent_mandi_prices(
            str(arguments.get("commodity", "")), state.strip(), arguments.get("district")
        )

    if name == "get_market_strategy":
        crop_name = arguments.get("crop_name")
        state = arguments.get("state")
        quantity = arguments.get("harvest_yield_quintals")
        distance = arguments.get("distance_to_mandi_km")
        missing = []
        if not isinstance(crop_name, str) or not crop_name.strip():
            missing.append("crop name")
        if not isinstance(state, str) or not state.strip():
            missing.append("state or union territory")
        if quantity is None:
            missing.append("expected harvest quantity in quintals")
        if distance is None:
            missing.append("distance to nearest mandi in kilometers")
        if missing:
            return {"needs_more_info": missing, "message": "Ask for the missing values before estimating market strategy."}
        request = EnhancedMarketRequest(
            crop_name=crop_name.strip(),
            state=state.strip(),
            harvest_yield_quintals=float(quantity),
            distance_to_mandi_km=float(distance),
        )
        return await evaluate_market_strategy(request)

    return {"error": "This agent tool is not available."}


def _persist_chatbot_record(user_query: str, language: str, bot_response: str) -> None:
    db = get_db()
    if not db:
        return
    try:
        db.add(ChatbotRecord(user_query=user_query, detected_language=language, bot_response=bot_response))
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"Chat history could not be saved: {exc}")
    finally:
        db.close()


@app.post("/api/v1/chatbot")
async def agronomic_chat_assistant(req: ChatbotRequest, background_tasks: BackgroundTasks):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")
    target_lang = detect_language(message.lower(), req.language)

    try:
        reply, selected_agent_model = await run_agri_agent(
            message=message,
            language=target_lang,
            history=[turn.model_dump() for turn in req.history],
            execute_tool=execute_agri_agent_tool,
        )
    except AgentConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AgentAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    background_tasks.add_task(_persist_chatbot_record, message, target_lang, reply)
    return {
        "user_query": message,
        "detected_language": target_lang,
        "bot_response": reply,
        "voice_lang": "te-IN" if target_lang == "te" else ("hi-IN" if target_lang == "hi" else ("ta-IN" if target_lang == "ta" else "en-US")),
        "agent_mode": "gemini",
        "agent_model": selected_agent_model,
    }


@app.post("/api/v1/speech")
async def generate_assistant_speech(req: SpeechRequest):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="Gemini speech is not configured. Set GEMINI_API_KEY in the backend environment.")

    language_names = {
        "te-IN": "Telugu",
        "ta-IN": "Tamil",
        "hi-IN": "Hindi",
        "en-US": "English",
    }
    speech_language = language_names.get(req.language, "the language of the provided text")
    configured_tts_model = os.getenv("GEMINI_TTS_MODEL", "").strip()
    default_tts_model = (
        "gemini-3.8-flash-tts"
        if req.language in {"te-IN", "ta-IN"}
        else "gemini-3.8-flash-lite-tts"
    )

    payload = {
        "model": configured_tts_model or default_tts_model,
        "input": [{
            "type": "user_input",
            "content": [{
                "type": "text",
                "text": req.text.strip(),
                "annotations": [{
                    "type": "speech_metadata",
                    "style": f"Speak clearly and naturally in {speech_language}.",
                }],
            }],
        }],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": [{"voice": "Kore"}]},
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            response = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/interactions",
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Gemini speech generation timed out. Try the audio player again.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Could not connect to Gemini speech generation.") from exc

    try:
        result = response.json()
    except ValueError:
        result = {}

    if not response.is_success:
        error = result.get("error") if isinstance(result, dict) else None
        detail = error.get("message") if isinstance(error, dict) else None
        if response.status_code in {401, 403}:
            raise HTTPException(status_code=502, detail="Gemini rejected the API key for speech generation.")
        if response.status_code in {429, 503}:
            raise HTTPException(status_code=503, detail="Gemini speech is temporarily busy or rate-limited. Try again shortly.")
        raise HTTPException(status_code=502, detail=detail or f"Gemini speech generation failed ({response.status_code}).")

    audio_data = ""
    for step in result.get("steps", []) if isinstance(result, dict) else []:
        if not isinstance(step, dict) or step.get("type") != "model_output":
            continue
        for item in step.get("content", []):
            if isinstance(item, dict) and item.get("type") == "audio" and item.get("data"):
                audio_data = item["data"]

    if not audio_data:
        raise HTTPException(status_code=502, detail="Gemini returned no speech audio. Try again.")

    try:
        audio_bytes = base64.b64decode(audio_data, validate=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Gemini returned invalid speech audio.") from exc

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


# Serve the static website and API from one origin for simple server deployment.
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
