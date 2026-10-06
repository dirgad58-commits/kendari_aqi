"""Memuat model XGBoost + scaler, membangun fitur, dan menghasilkan prediksi."""
import json
import math
import pickle
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DATA_FILE = ROOT / "data" / "kendari_aqi_pm25_dataset.csv"

def _load(path: Path):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


@st.cache_resource(show_spinner="Memuat model...")
def load_artifacts():
    """Mengembalikan (model, scaler, metadata, error). Error berisi pesan jika gagal."""
    try:
        meta = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
        model = _load(MODEL_DIR / "xgboost_pm25_model.pkl")
        scaler = _load(MODEL_DIR / "scaler.pkl")
        return model, scaler, meta, None
    except Exception as exc:  # noqa: BLE001
        return None, None, None, f"{type(exc).__name__}: {exc}"


def build_features(d: date, *, temperature_c, humidity_pct, rainfall_mm, wind_speed_kmh,
                   lag1, lag3, lag7, roll7, is_holiday=False) -> dict:
    month_angle = 2 * math.pi * d.month / 12
    # Sama dengan notebook: day_sin/day_cos memakai hari dalam pekan (0-6) / 7
    day_angle = 2 * math.pi * d.weekday() / 7.0
    return {
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "rainfall_mm": rainfall_mm,
        "wind_speed_kmh": wind_speed_kmh,
        "rain_x_wind": rainfall_mm * wind_speed_kmh,
        "is_weekend": int(d.weekday() >= 5),
        "is_holiday": int(is_holiday),
        "day_of_week": d.weekday(),
        "month_sin": math.sin(month_angle),
        "month_cos": math.cos(month_angle),
        "day_sin": math.sin(day_angle),
        "day_cos": math.cos(day_angle),
        "pm25_lag_1d": lag1,
        "pm25_lag_3d": lag3,
        "pm25_lag_7d": lag7,
        "pm25_rolling_mean_7d": roll7,
    }


def predict_pm25(model, scaler, features: list, rows: list) -> list:
    """rows: list of dict dari build_features. Mengembalikan list PM2.5 (µg/m³)."""
    X = pd.DataFrame(rows)[features]
    Xs = pd.DataFrame(scaler.transform(X), columns=features)
    preds = model.predict(Xs)
    return [max(0.0, float(p)) for p in preds]


def feature_importance(model, features: list):
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return None
    return sorted(zip(features, [float(v) for v in values]), key=lambda x: x[1], reverse=True)


@st.cache_data(show_spinner=False)
def load_history():
    """Membaca dataset hasil ETL bila tersedia; None jika tidak ada."""
    if not DATA_FILE.exists():
        return None
    df = pd.read_csv(DATA_FILE)
    if "date" not in df.columns or "pm25" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def defaults_from_history(df):
    """Nilai awal lag dari data terakhir (untuk mengisi sidebar)."""
    if df is None or len(df) < 7:
        return None
    s = df["pm25"].dropna().tolist()
    if len(s) < 7:
        return None
    return {
        "lag1": round(s[-1], 1),
        "lag3": round(s[-3], 1),
        "lag7": round(s[-7], 1),
        "roll7": round(sum(s[-7:]) / 7, 1),
    }
