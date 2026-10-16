import os
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, Float, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DEFAULT_DB_PATH = Path(__file__).with_name("smart_agriguide.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}").strip()
engine_options = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite:"):
    engine_options["connect_args"] = {"check_same_thread": False}

try:
    engine = create_engine(DATABASE_URL, **engine_options)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception:
    engine = None
    SessionLocal = None

Base = declarative_base()

# -------------------------------------------------------------
# TABLE 1: Crop Suitability Recommendations
# -------------------------------------------------------------
class CropRecord(Base):
    __tablename__ = "crop_recommendations"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    nitrogen = Column(Float)
    phosphorus = Column(Float)
    potassium = Column(Float)
    temperature = Column(Float)
    humidity = Column(Float)
    ph = Column(Float)
    rainfall = Column(Float)
    predicted_crop = Column(String(100))
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

# -------------------------------------------------------------
# TABLE 2: Foliar Pathology Diagnoses
# -------------------------------------------------------------
class LeafRecord(Base):
    __tablename__ = "leaf_diagnostics"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    diagnosis = Column(String(150))
    pathogen_type = Column(String(150))
    severity = Column(String(50))
    confidence_score = Column(Float)
    recommended_treatment = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# -------------------------------------------------------------
# TABLE 3: Smart Hydrological Telemetry
# -------------------------------------------------------------
class IrrigationRecord(Base):
    __tablename__ = "irrigation_telemetry"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    soil_moisture_pct = Column(Float)
    field_capacity_pct = Column(Float)
    precipitation_probability_pct = Column(Float)
    forecast_rainfall_mm = Column(Float)
    recommended_irrigation_depth_mm = Column(Float)
    advisory_status = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# -------------------------------------------------------------
# TABLE 4: Market Strategy Inquiries
# -------------------------------------------------------------
class MarketRecord(Base):
    __tablename__ = "market_inquiries"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    crop_name = Column(String(100))
    harvest_yield_quintals = Column(Float)
    distance_to_mandi_km = Column(Float)
    current_year_rate = Column(Float)
    previous_year_rate = Column(Float)
    optimal_decision = Column(String(50))
    best_channel = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

# -------------------------------------------------------------
# TABLE 5: Multilingual Chatbot & Voice Conversations
# -------------------------------------------------------------
class ChatbotRecord(Base):
    __tablename__ = "chatbot_conversations"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_query = Column(Text)
    detected_language = Column(String(20))
    bot_response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    if engine is not None:
        try:
            Base.metadata.create_all(bind=engine)
            print("SQL database tables verified and initialized successfully.")
        except Exception as err:
            print(f"Warning: Could not initialize the configured SQL database: {err}")
