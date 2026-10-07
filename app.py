"""Dashboard prediksi PM2.5 & Tingkatan Kualitas Udara Kota Kendari (XGBoost) - Streamlit."""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.aqi import CATEGORIES, WHO_DAILY_LIMIT, get_category, pm25_to_aqi
from utils.predictor import (build_features, defaults_from_history, extra_default, extra_features,
                             feature_importance, load_artifacts, load_history, predict_pm25)

ROOT = Path(__file__).parent

st.set_page_config(page_title="Kualitas Udara Kendari", page_icon="🌫️", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------ helpers
st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)


def html(s: str):
    """Render HTML tanpa indentasi (agar tidak dibaca sebagai blok kode markdown)."""
    st.markdown("\n".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def section(title: str, desc: str = ""):
    html(f'<div class="section-head"><h2>{title}</h2>{f"<p>{desc}</p>" if desc else ""}</div>')


FEATURE_LABELS = {
    "temperature_c": "Suhu udara", "humidity_pct": "Kelembapan", "rainfall_mm": "Curah hujan",
    "wind_speed_kmh": "Kecepatan angin", "rain_x_wind": "Hujan × angin", "is_weekend": "Akhir pekan",
    "is_holiday": "Hari libur nasional", "day_of_week": "Hari dalam pekan",
    "month_sin": "Bulan (sinus)", "month_cos": "Bulan (kosinus)",
    "day_sin": "Hari dalam pekan (sinus)", "day_cos": "Hari dalam pekan (kosinus)",
    "pm25_lag_1d": "PM2.5 1 hari lalu", "pm25_lag_3d": "PM2.5 3 hari lalu",
    "pm25_lag_7d": "PM2.5 7 hari lalu", "pm25_rolling_mean_7d": "Rata-rata PM2.5 7 hari",
    "rain_lag_1d": "Hujan kemarin (mm)", "rain_sum_3d": "Total hujan 3 hari (mm)",
    "wind_lag_1d": "Angin kemarin (km/jam)", "wind_mean_3d": "Rata-rata angin 3 hari (km/jam)",
    "humidity_lag_1d": "Kelembapan kemarin (%)", "pm10_lag_1d": "PM10 kemarin",
    "no2_lag_1d": "NO2 kemarin", "co_lag_1d": "CO kemarin", "o3_lag_1d": "O3 kemarin",
    "doy_sin": "Musim (sinus)", "doy_cos": "Musim (kosinus)",
}

# ------------------------------------------------------------------ template definisi
TEMPLATES = {
    "🌿 Normal / Bersih (Tipikal Kendari)": {
        "badge": "🟢 Normal / Bersih",
        "desc": "Kondisi umum harian pesisir Kendari: cuaca tropis normal, hembusan angin stabil, dan kualitas udara tergolong bersih.",
        "temp": 27.5, "hum": 82.0, "rain": 2.0, "wind": 6.0,
        "lag1": 11.0, "lag3": 10.5, "lag7": 11.2, "roll7": 11.0,
        "extra": {
            "rain_lag_1d": 2.0, "rain_sum_3d": 6.0, "wind_lag_1d": 5.5, "wind_mean_3d": 5.8, "humidity_lag_1d": 82.0,
        }
    },
    "🌧️ Hujan Lebat (Wet Scavenging / Partikel Tercuci)": {
        "badge": "🌧️ Hujan Lebat",
        "desc": "Curah hujan tinggi berturut-turut meluruhkan partikulat udara secara signifikan (wet deposition), membuat konsentrasi PM2.5 turun tajam.",
        "temp": 24.5, "hum": 95.0, "rain": 45.0, "wind": 8.5,
        "lag1": 8.0, "lag3": 8.5, "lag7": 10.0, "roll7": 9.0,
        "extra": {
            "rain_lag_1d": 25.0, "rain_sum_3d": 70.0, "wind_lag_1d": 8.0, "wind_mean_3d": 8.2, "humidity_lag_1d": 94.0,
        }
    },
    "☀️ Kemarau Terik & Udara Kering (Stagnan)": {
        "badge": "☀️ Kemarau Kering",
        "desc": "Cuaca panas terik berhari-hari tanpa hujan, kelembapan menurun dan angin lemah sehingga partikulat debu terakumulasi di udara.",
        "temp": 33.5, "hum": 66.0, "rain": 0.0, "wind": 2.5,
        "lag1": 22.0, "lag3": 19.5, "lag7": 17.0, "roll7": 19.0,
        "extra": {
            "rain_lag_1d": 0.0, "rain_sum_3d": 0.0, "wind_lag_1d": 3.0, "wind_mean_3d": 2.8, "humidity_lag_1d": 68.0,
        }
    },
    "🏭 Polusi Udara Tinggi / Asap (Kritis)": {
        "badge": "🔴 Polusi Tinggi",
        "desc": "Kondisi udara kritis: angin tenang (inversi/stagnan), tanpa hujan, dan riwayat partikulat PM2.5 sangat tinggi (misal akibat asap/karhutla).",
        "temp": 32.5, "hum": 58.0, "rain": 0.0, "wind": 1.2,
        "lag1": 55.0, "lag3": 48.0, "lag7": 40.0, "roll7": 46.0,
        "extra": {
            "rain_lag_1d": 0.0, "rain_sum_3d": 0.0, "wind_lag_1d": 1.5, "wind_mean_3d": 1.4, "humidity_lag_1d": 60.0,
        }
    },
    "💨 Angin Kencang Pesisir (Dispersi Cepat)": {
        "badge": "💨 Angin Kencang",
        "desc": "Kecepatan angin tinggi dari arah pesisir Teluk Kendari mempercepat sirkulasi dan menyebarkan partikel polusi ke luar daratan.",
        "temp": 26.5, "hum": 75.0, "rain": 1.0, "wind": 20.0,
        "lag1": 12.0, "lag3": 12.5, "lag7": 13.0, "roll7": 12.2,
        "extra": {
            "rain_lag_1d": 1.0, "rain_sum_3d": 3.0, "wind_lag_1d": 18.0, "wind_mean_3d": 19.0, "humidity_lag_1d": 76.0,
        }
    },
    "⚙️ Kustom (Input Bebas)": {
        "badge": "⚙️ Manual / Kustom",
        "desc": "Atur seluruh variabel suhu, kelembapan, angin, hujan, dan lag secara mandiri sesuai kondisi aktual yang ingin Anda uji.",
        "temp": 27.0, "hum": 86.0, "rain": 5.0, "wind": 5.4,
        "lag1": 11.9, "lag3": 11.9, "lag7": 11.9, "roll7": 11.9,
        "extra": {
            "rain_lag_1d": 5.0, "rain_sum_3d": 15.0, "wind_lag_1d": 5.4, "wind_mean_3d": 5.4, "humidity_lag_1d": 86.0,
        }
    }
}

# ------------------------------------------------------------------ load model
model, scaler, meta, load_error = load_artifacts()
if load_error:
    html(f"""<div class="note warn"><b>Model belum bisa dimuat.</b><br>
    Pastikan folder <code>models/</code> berisi <code>xgboost_pm25_model.pkl</code>,
    <code>scaler.pkl</code>, dan <code>model_metadata.json</code>, serta
    <code>xgboost</code> ada di <code>requirements.txt</code>.<br>
    Detail: <code>{load_error}</code></div>""")
    st.stop()

FEATURES = meta["features"]
LEAK_FIXED = bool(meta.get("rolling_mean_excludes_target"))
LEAK_NOTE = "" if LEAK_FIXED else (
    "<p>Model ini dilatih dengan fitur rata-rata 7 hari yang ikut memuat PM2.5 hari target, "
    "sehingga metrik evaluasi cenderung terlalu optimistis. Latih ulang dengan notebook versi perbaikan.</p>")
history = load_history()
hist_defaults = defaults_from_history(history)
extras = extra_features(FEATURES)

# ------------------------------------------------------------------ template apply helper
def apply_template_values(key_name):
    if key_name in TEMPLATES and key_name != "⚙️ Kustom (Input Bebas)":
        t = TEMPLATES[key_name]
        st.session_state["temp_input"] = float(t["temp"])
        st.session_state["hum_input"] = float(t["hum"])
        st.session_state["rain_input"] = float(t["rain"])
        st.session_state["wind_input"] = float(t["wind"])
        st.session_state["lag1_input"] = float(t["lag1"])
        st.session_state["lag3_input"] = float(t["lag3"])
        st.session_state["lag7_input"] = float(t["lag7"])
        st.session_state["roll7_input"] = float(t["roll7"])
        for k, v in t.get("extra", {}).items():
            st.session_state[f"extra_{k}"] = float(v)

# Inisialisasi session state pertama kali
if "template_choice" not in st.session_state:
    st.session_state["template_choice"] = "🌿 Normal / Bersih (Tipikal Kendari)"
    apply_template_values("🌿 Normal / Bersih (Tipikal Kendari)")

if "temp_input" not in st.session_state:
    st.session_state["temp_input"] = 27.5
if "hum_input" not in st.session_state:
    st.session_state["hum_input"] = 82.0
if "rain_input" not in st.session_state:
    st.session_state["rain_input"] = 2.0
if "wind_input" not in st.session_state:
    st.session_state["wind_input"] = 6.0
if "lag1_input" not in st.session_state:
    d = hist_defaults or {"lag1": 11.0, "lag3": 10.5, "lag7": 11.2, "roll7": 11.0}
    st.session_state["lag1_input"] = float(d["lag1"])
    st.session_state["lag3_input"] = float(d["lag3"])
    st.session_state["lag7_input"] = float(d["lag7"])
    st.session_state["roll7_input"] = float(d["roll7"])
for name in extras:
    if f"extra_{name}" not in st.session_state:
        st.session_state[f"extra_{name}"] = float(extra_default(name, FEATURES, scaler, history))

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    html('<p class="side-title">Pengaturan Kalender</p>'
         '<p class="side-note">Atur tanggal target prediksi dan status hari libur nasional Kota Kendari.</p>')
    target_date = st.date_input("Tanggal prediksi", value=date.today() + timedelta(days=1), key="target_date_val")
    is_holiday = st.checkbox("Hari libur nasional", value=False, key="is_holiday_val")

    st.markdown("---")
    html('<p class="side-title">Status Model XGBoost</p>')
    st.markdown(f"- **Algoritma**: `{meta.get('algorithm', 'XGBoost')}`")
    st.markdown(f"- **Jumlah Fitur**: `{len(FEATURES)} fitur`")
    st.markdown(f"- **MAE Pengujian**: `{meta['metrics']['mae']:.2f} µg/m³`")
    st.markdown(f"- **R² Score**: `{meta['metrics']['r2']:.2f}`")
    st.markdown(f"- **Rolling Window Leak**: `{'Bebas Leakage ✅' if LEAK_FIXED else 'Versi Lama'}`")

    st.markdown("---")
    if st.button("🔄 Reset Sesi & Prediksi", use_container_width=True):
        if "prediction_result" in st.session_state:
            del st.session_state["prediction_result"]
        st.session_state["template_choice"] = "🌿 Normal / Bersih (Tipikal Kendari)"
        apply_template_values("🌿 Normal / Bersih (Tipikal Kendari)")
        st.rerun()

# ------------------------------------------------------------------ hero
spectrum = "".join(f'<span style="background:{c.color}"></span>' for c in CATEGORIES)
trained = meta.get("created_at", "-")
html(f"""
<div class="hero">
  <h1>Perkiraan Kualitas Udara Kota Kendari</h1>
  <p>Model XGBoost memperkirakan konsentrasi PM2.5 harian dari data cuaca dan riwayat polusi,
  lalu memetakan ke dalam 6 tingkatan Indeks Standar Kualitas Udara (US EPA AQI 2024).</p>
  <div class="spectrum">{spectrum}</div>
  <div class="spectrum-labels"><span>Tingkat 1: Baik</span><span>Tingkat 6: Berbahaya</span></div>
  <div class="chips"><span class="chip">{meta.get('algorithm', 'XGBoost')}</span>
  <span class="chip">{len(FEATURES)} fitur</span><span class="chip">Dilatih {trained[:10]}</span>
  <span class="chip">{"Leak-Free Rolling ✅" if LEAK_FIXED else "Legacy"}</span></div>
</div>
""")

tab_pred, tab_sim, tab_perf, tab_hist, tab_about = st.tabs(
    ["🔮 Prediksi & Tingkatan", "Simulasi cuaca", "Performa model", "Data historis", "Tentang"])


def scale_html(aqi: int) -> str:
    segs = [(c, c.i_high - c.i_low + (1 if c.i_low else 0)) for c in CATEGORIES[:5]]
    bar = "".join(f'<span style="background:{c.color};width:{w / 300 * 100:.2f}%"></span>' for c, w in segs)
    pos = min(max(0, aqi), 300) / 300 * 100
    return (f'<div class="scale"><div class="scale-bar">{bar}</div>'
            f'<div class="scale-pin" style="left:{pos:.1f}%"></div>'
            '<div class="scale-ticks"><span>0 (Baik)</span><span>50</span><span>100</span>'
            '<span>150</span><span>200</span><span>300+ (Bahaya)</span></div></div>')


def render_levels_table(active_level: int = 0):
    rows_html = ""
    for c in CATEGORIES:
        is_active = (c.level == active_level)
        active_class = "active" if is_active else ""
        border_style = f"border-left: 6px solid {c.color};" if is_active else f"border-left: 3px solid {c.color};"
        badge_now = f'<span style="background:{c.color};color:#fff;padding:2px 8px;border-radius:99px;font-size:0.75rem;font-weight:700;margin-left:6px;">👉 AKTIF</span>' if is_active else ""
        rows_html += f"""
        <div class="level-card {active_class}" style="{border_style}">
          <div>
            <div style="font-weight:700; font-size:0.95rem; color:var(--ink);">
              <span class="level-pill" style="background:{c.color}">Tingkat {c.level}</span>
              <span style="margin-left:6px;">{c.name}</span>
              {badge_now}
            </div>
            <div style="color:var(--ink-soft); font-size:0.83rem; margin-top:2px;">{c.advice}</div>
          </div>
          <div style="text-align:right; min-width:130px;">
            <div style="font-weight:700; font-size:0.9rem; color:{c.color};">AQI {c.i_low} – {c.i_high}</div>
            <div style="font-size:0.8rem; color:var(--ink-soft);">{c.c_low} – {c.c_high} µg/m³</div>
          </div>
        </div>
        """
    return rows_html


# ================================================================== TAB 1: PREDIKSI
with tab_pred:
    section("1. Pilih Template Kondisi (Preset Siap Pakai)",
            "Klik tombol template di bawah untuk memuat skenario cuaca & riwayat secara instan, atau pilih dari menu dropdown.")

    # Tombol preset cepat
    cols_preset = st.columns(len(TEMPLATES) - 1)
    preset_names = [k for k in TEMPLATES.keys() if k != "⚙️ Kustom (Input Bebas)"]
    for col, p_name in zip(cols_preset, preset_names):
        short_label = p_name.split(" / ")[0].split(" (")[0]
        if col.button(short_label, use_container_width=True, key=f"quick_btn_{short_label}"):
            st.session_state["template_choice"] = p_name
            apply_template_values(p_name)
            st.rerun()

    c_tmpl1, c_tmpl2 = st.columns([1.5, 2.5], gap="medium")
    with c_tmpl1:
        selected_tmpl = st.selectbox(
            "Pilihan Template Skenario:",
            list(TEMPLATES.keys()),
            key="template_choice",
            on_change=lambda: apply_template_values(st.session_state.template_choice)
        )
    with c_tmpl2:
        curr_t = TEMPLATES[selected_tmpl]
        html(f"""
        <div class="template-card" style="margin-top: 1.6rem;">
          <div>
            <span class="template-badge">{curr_t['badge']}</span>
            <div style="font-size:0.88rem; color:var(--ink-soft); margin-top:0.3rem;">{curr_t['desc']}</div>
          </div>
        </div>
        """)

    section("2. Parameter Masukan (Dapat Disesuaikan)",
            "Nilai di bawah ini telah terisi otomatis dari template yang Anda pilih. Anda dapat mengubahnya jika diperlukan.")

    col_w, col_lag = st.columns(2, gap="large")

    with col_w:
        with st.container(border=True):
            st.markdown("#### ⛅ Parameter Cuaca Hari Prediksi")
            temp = st.slider("Suhu udara (°C)", 20.0, 42.0, step=0.1, key="temp_input")
            hum = st.slider("Kelembapan udara (%)", 30.0, 100.0, step=0.5, key="hum_input")
            rain = st.slider("Curah hujan harian (mm)", 0.0, 150.0, step=0.5, key="rain_input")
            wind = st.slider("Kecepatan angin (km/jam)", 0.0, 40.0, step=0.1, key="wind_input")

    with col_lag:
        with st.container(border=True):
            st.markdown("#### 📊 Parameter Riwayat Polusi & Lag Cuaca")
            lag1 = st.number_input("PM2.5 kemarin (1 hari lalu, µg/m³)", 0.0, 500.0, step=0.1, key="lag1_input")
            c_sub1, c_sub2 = st.columns(2)
            with c_sub1:
                lag3 = st.number_input("PM2.5 3 hari lalu", 0.0, 500.0, step=0.1, key="lag3_input")
            with c_sub2:
                lag7 = st.number_input("PM2.5 7 hari lalu", 0.0, 500.0, step=0.1, key="lag7_input")
            roll7 = st.number_input("Rata-rata PM2.5 7 hari terakhir (µg/m³)", 0.0, 500.0, step=0.1, key="roll7_input")

            extra_vals = {}
            if extras:
                with st.expander("🛠️ Fitur Lag Cuaca Pendukung (Model 21 Fitur)", expanded=False):
                    st.caption("Nilai otomatis terisi dari template. Sesuaikan jika diperlukan:")
                    for name in extras:
                        extra_vals[name] = st.number_input(
                            FEATURE_LABELS.get(name, name), 0.0, 100000.0,
                            step=0.1, key=f"extra_{name}")
            else:
                extra_vals = {}

    # ------------------ TOMBOL EKSEKUSI PREDIKSI ("PENCET")
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    c_btn1, c_btn2 = st.columns([3, 1])
    with c_btn1:
        predict_clicked = st.button("🔮 HITUNG PREDIKSI & TINGKATAN AQI (PENCET DI SINI)",
                                    type="primary", use_container_width=True)
    with c_btn2:
        reset_clicked = st.button("🔄 Bersihkan Hasil", use_container_width=True)

    if reset_clicked:
        if "prediction_result" in st.session_state:
            del st.session_state["prediction_result"]
        st.rerun()

    if predict_clicked:
        with st.spinner("Menghitung perkiraan PM2.5 dan menentukan tingkatan AQI..."):
            weather = dict(temperature_c=temp, humidity_pct=hum, rainfall_mm=rain, wind_speed_kmh=wind)
            row = build_features(target_date, **weather, lag1=lag1, lag3=lag3, lag7=lag7,
                                 roll7=roll7, is_holiday=is_holiday, extra=extra_vals)
            pred_pm = predict_pm25(model, scaler, FEATURES, [row])[0]
            pred_aqi = pm25_to_aqi(pred_pm)
            pred_cat = get_category(pred_aqi)
            st.session_state["prediction_result"] = {
                "pm": pred_pm,
                "aqi": pred_aqi,
                "cat": pred_cat,
                "weather": weather,
                "template": st.session_state.get("template_choice", "Kustom"),
                "target_date": target_date
            }
            st.toast(f"Prediksi berhasil! Tingkat {pred_cat.level}: {pred_cat.name} (AQI {pred_aqi})", icon="✅")

    # ------------------ HASIL PREDIKSI ATAU PANDUAN TINGKATAN
    res = st.session_state.get("prediction_result")
    if res is not None:
        pm = res["pm"]
        aqi = res["aqi"]
        cat = res["cat"]
        diff = (pm - WHO_DAILY_LIMIT) / WHO_DAILY_LIMIT * 100
        who_txt = (f"{abs(diff):.0f}% di atas batas aman WHO" if diff > 0 else f"{abs(diff):.0f}% di bawah batas aman WHO")

        section("3. Hasil Prediksi Kualitas Udara",
                f"Skenario: {res['template']} · Tanggal Prediksi: {res['target_date'].strftime('%d %B %Y')}")

        col_res_left, col_res_right = st.columns([1.15, 1.25], gap="large")

        with col_res_left:
            html(f"""
            <div class="result" style="box-shadow: 0 4px 18px rgba(0,0,0,0.07); border: 2px solid {cat.color};">
              <div class="result-band" style="background:{cat.color}">
                <div style="font-size:0.85rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; opacity:0.95;">
                  STATUS TINGKATAN KUALITAS UDARA
                </div>
                <div class="cat" style="font-size:1.6rem; margin-top:0.25rem;">Tingkat {cat.level}: {cat.name}</div>
                <div class="aqi">{aqi}</div>
                <div class="unit">Indeks Standar AQI US EPA 2024</div>
              </div>
              <div class="result-body">
                <p style="font-size:1.02rem; font-weight:500; margin-bottom:1rem;">{cat.advice}</p>
                {scale_html(aqi)}
                <div class="facts" style="margin-top:1.2rem">
                  <div class="fact">
                    <b>{pm:.1f} µg/m³</b>
                    <span>Konsentrasi PM2.5</span>
                  </div>
                  <div class="fact">
                    <b style="color:{'#d64545' if diff > 0 else '#2e9e5b'}">{abs(diff):.0f}% {'DI ATAS' if diff > 0 else 'DI BAWAH'}</b>
                    <span>Batas WHO 24 Jam ({WHO_DAILY_LIMIT:.0f} µg/m³)</span>
                  </div>
                </div>
              </div>
            </div>
            """)

        with col_res_right:
            st.markdown("#### 📋 Posisi pada Standar Tingkatan AQI")
            html(render_levels_table(active_level=cat.level))

    else:
        section("3. Tingkatan Standar Kualitas Udara (US EPA AQI 2024)",
                "Pilih template atau sesuaikan input di atas, lalu pencet tombol di atas untuk memunculkan hasil prediksi.")
        html("""
        <div class="note" style="border-left-color: var(--accent); background: var(--accent-soft); padding: 1rem 1.2rem; margin-bottom: 1.2rem;">
          <h4 style="margin: 0 0 0.3rem; color: var(--accent);">👉 Belum Ada Prediksi Dihitung</h4>
          <p style="margin: 0; line-height: 1.5;">Tekan tombol <b>"🔮 HITUNG PREDIKSI & TINGKATAN AQI (PENCET DI SINI)"</b> di atas untuk menjalankan model XGBoost dan memunculkan hasil tingkat kualitas udara.</p>
        </div>
        """)
        html(render_levels_table(active_level=0))

# ================================================================== TAB 2: SIMULASI CUACA
with tab_sim:
    section("Bagaimana jika cuacanya berubah?",
            "Empat skenario dihitung dengan kondisi riwayat polusi saat ini; hanya variabel cuaca yang diubah.")
    scenarios = {
        "Kemarau ekstrem": ("Skenario What-If: cuaca panas terik, tanpa hujan, angin pelan 2 km/jam.",
                            dict(temperature_c=33.5, humidity_pct=85.0, rainfall_mm=0.0, wind_speed_kmh=2.0)),
        "Cuaca normal": ("Mendekati nilai rata-rata pesisir Kendari.",
                         dict(temperature_c=27.0, humidity_pct=86.0, rainfall_mm=5.0, wind_speed_kmh=5.4)),
        "Angin kencang": ("Angin pesisir kencang menyebarkan partikel polutan.",
                          dict(temperature_c=27.0, humidity_pct=82.0, rainfall_mm=1.0, wind_speed_kmh=12.0)),
        "Hujan lebat": ("Presipitasi lebat meluruhkan partikel dari atmosfer (scavenging).",
                        dict(temperature_c=24.5, humidity_pct=96.0, rainfall_mm=45.0, wind_speed_kmh=7.0)),
    }
    rows = [build_features(target_date, **w, lag1=lag1, lag3=lag3, lag7=lag7, roll7=roll7,
                           is_holiday=is_holiday, extra=extra_vals) for _, w in scenarios.values()]
    preds = predict_pm25(model, scaler, FEATURES, rows)

    cards = ""
    for (name, (desc, w)), p in zip(scenarios.items(), preds):
        a = pm25_to_aqi(p)
        c = get_category(a)
        cards += f"""
        <div class="scen" style="border-top-color:{c.color}">
          <h4>{name}</h4>
          <div class="desc">{desc}</div>
          <div class="big" style="color:{c.color}">{p:.1f}<span style="font-size:1rem;font-weight:600"> µg/m³</span></div>
          <div class="small">Tingkat {c.level}: {c.name} (AQI {a})</div>
          <div class="small" style="margin-top:.5rem">Hujan {w['rainfall_mm']:.0f} mm · angin {w['wind_speed_kmh']:.1f} km/jam</div>
        </div>"""
    html(f'<div class="scen-grid">{cards}</div>')

    spread = max(preds) - min(preds)
    html(f"""<div class="note"><b>Selisih antar skenario: {spread:.2f} µg/m³.</b>
    Jika selisihnya kecil, model lebih banyak dipengaruhi oleh riwayat PM2.5 dan musiman
    daripada cuaca sesaat. Anda dapat menguji template lain di tab Prediksi.</div>""")

# ================================================================== TAB 3: PERFORMA MODEL
with tab_perf:
    m = meta["metrics"]
    section("Seberapa akurat model XGBoost ini?", "Hasil evaluasi pada data uji (20% data terakhir, urutan kronologis tanpa shuffle).")
    html(f"""
    <div class="metrics">
      <div class="metric"><div class="label">MAE</div><div class="value">{m['mae']:.2f}</div>
        <div class="hint">Rata-rata selisih absolut prediksi dan aktual, dalam µg/m³.</div></div>
      <div class="metric"><div class="label">RMSE</div><div class="value">{m['rmse']:.2f}</div>
        <div class="hint">Sensitif terhadap kesalahan ekstrem/outlier besar.</div></div>
      <div class="metric"><div class="label">R² Score</div><div class="value">{m['r2']:.2f}</div>
        <div class="hint">Model menjelaskan sekitar {m['r2'] * 100:.0f}% variasi PM2.5 pada data uji.</div></div>
    </div>
    """)
    if m["r2"] < 0.6:
        html(f"""<div class="note warn"><b>Catatan Evaluasi:</b>
        R² {m['r2']:.2f} merefleksikan dinamika cuaca tropis pesisir di mana dispersi harian sangat dipengaruhi oleh faktor mikroklimat lokal.
        Prediksi sangat andal sebagai indikator tingkatan tren kualitas udara harian.</div>""")

    bl = meta.get("baseline")
    if bl:
        better = m["r2"] > bl["persistence_r2"]
        html(f"""<div class="note{'' if better else ' warn'}"><b>Pembanding Model Dasar (Persistence Baseline):</b>
        Menebak nilai PM2.5 sama dengan hari kemarin menghasilkan MAE {bl['persistence_mae']:.2f} dan R² {bl['persistence_r2']:.2f}.
        Model XGBoost {'berhasil mengungguli baseline tersebut secara konsisten ✅' if better else 'belum mengungguli baseline tersebut'}.</div>""")
    if not LEAK_FIXED:
        html("""<div class="note warn"><b>Model versi lama.</b> Fitur rolling mean pada pelatihan memuat
        target leakage. Gunakan model hasil notebook terbaru.</div>""")

    section("Fitur Paling Berpengaruh (Feature Importance)")
    imp = feature_importance(model, FEATURES)
    if imp:
        top = imp[:10]
        mx = top[0][1] or 1
        bars = "".join(
            f'<div class="bar-row"><span>{FEATURE_LABELS.get(f, f)}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{v / mx * 100:.1f}%"></div></div>'
            f'<span class="bar-val">{v * 100:.1f}%</span></div>' for f, v in top)
        html(f'<div class="panel"><div class="sub">10 Fitur dengan kontribusi kepentingan tertinggi pada pohon keputusan XGBoost.</div>'
             f'<div class="bars">{bars}</div></div>')
    else:
        items = "".join(f"<li>{FEATURE_LABELS.get(f, f)}</li>" for f in meta.get("top_features", []))
        html(f'<div class="panel"><div class="sub">Fitur teratas dari metadata model:</div><ol>{items}</ol></div>')

# ================================================================== TAB 4: DATA HISTORIS
with tab_hist:
    section("Riwayat PM2.5 Kota Kendari", "Dari dataset hasil pipeline ETL Apache Airflow.")
    if history is None:
        html("""<div class="note">Dataset riwayat belum ditemukan di <code>data/kendari_aqi_pm25_dataset.csv</code>.
        Pastikan file dataset dari hasil DAG Airflow ditaruh pada direktori repo.</div>""")
    else:
        s = history["pm25"].dropna()
        over = (s > WHO_DAILY_LIMIT).mean() * 100
        html(f"""
        <div class="metrics">
          <div class="metric"><div class="label">Total Pengamatan</div><div class="value">{len(s):,}</div>
            <div class="hint">{history['date'].min():%d %b %Y} – {history['date'].max():%d %b %Y}</div></div>
          <div class="metric"><div class="label">Rata-rata PM2.5</div><div class="value">{s.mean():.1f}</div>
            <div class="hint">µg/m³</div></div>
          <div class="metric"><div class="label">Hari di Atas Batas WHO</div><div class="value">{over:.0f}%</div>
            <div class="hint">Batas harian {WHO_DAILY_LIMIT:.0f} µg/m³</div></div>
        </div>""")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history["date"], y=history["pm25"], mode="lines",
                                 line=dict(color="#1f6f78", width=1.6), name="PM2.5 Aktual"))
        fig.add_hline(y=WHO_DAILY_LIMIT, line_dash="dash", line_color="#d64545",
                      annotation_text="Batas WHO (15 µg/m³)", annotation_position="top left")
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="#ffffff", font=dict(family="Source Sans 3", color="#12303a"),
                          yaxis_title="PM2.5 (µg/m³)", showlegend=False,
                          xaxis=dict(gridcolor="#eef2f1"), yaxis=dict(gridcolor="#eef2f1"))
        st.plotly_chart(fig, use_container_width=True)

        monthly = history.assign(bulan=history["date"].dt.month).groupby("bulan")["pm25"].mean()
        names = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
        fig2 = go.Figure(go.Bar(x=[names[i - 1] for i in monthly.index], y=monthly.values,
                                marker_color="#1f6f78"))
        fig2.add_hline(y=WHO_DAILY_LIMIT, line_dash="dash", line_color="#d64545")
        fig2.update_layout(title="Rata-rata Konsentrasi PM2.5 per Bulan (µg/m³)", height=300,
                           margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="#ffffff", font=dict(family="Source Sans 3", color="#12303a"),
                           yaxis_title="µg/m³", yaxis=dict(gridcolor="#eef2f1"))
        st.plotly_chart(fig2, use_container_width=True)

# ================================================================== TAB 5: TENTANG
with tab_about:
    section("Tentang Dashboard Kualitas Udara")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        html("""<div class="panel"><h3>Arsitektur & Alur Data</h3>
        <div class="sub">Pipeline otomatis dari sumber data hingga inferensi.</div>
        <p><b>1. Ekstraksi Otomatis:</b> Open-Meteo API (polutan udara), NASA POWER (parameter meteorologi), dan Nager.Date (hari libur nasional) diekstraksi via Apache Airflow.</p>
        <p><b>2. Transformasi & Feature Engineering:</b> Penggabungan lintas tabel berdasarkan tanggal, rekayasa lag (1d, 3d, 7d), rolling window 7d tanpa target leakage, serta sin/cos encoding musiman.</p>
        <p><b>3. Model Machine Learning:</b> Algoritma XGBoost Regressor dilatih menggunakan 21 fitur prediktif dengan pembagian kronologis (80% train, 20% test).</p>
        <p><b>4. Konversi Standar:</b> Output regresi PM2.5 dikonversi ke Indeks AQI US EPA (revisi 2024) menggunakan formula piecewise linear interpolation.</p></div>""")
    with c2:
        html(f"""<div class="panel"><h3>Catatan & Pedoman Penggunaan</h3>
        <div class="sub">Hal penting bagi pengguna aplikasi.</div>
        <p><b>Perkiraan Edukatif:</b> Nilai hasil model merupakan prediksi estimasi berbasis data satelit/reanalisis dan bukan pengganti stasiun pemantau resmi KLHK.</p>
        <p><b>Template Kondisi:</b> Gunakan preset skenario yang tersedia untuk menyimulasikan berbagai kondisi ekstrem (hujan deras, kemarau, maupun polusi tinggi).</p>
        <p><b>Standar Kesehatan:</b> Pedoman WHO menetapkan batas konsentrasi rata-rata 24 jam sebesar 15 µg/m³ sebagai ambang batas aman bagi kesehatan manusia.</p></div>""")
    section("Daftar 21 Fitur yang Digunakan Model")
    chips = "".join(f'<span class="chip">{FEATURE_LABELS.get(f, f)}</span>' for f in FEATURES)
    html(f'<div class="chips" style="margin-top:0">{chips}</div>')
