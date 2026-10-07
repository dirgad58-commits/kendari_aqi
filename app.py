"""Dashboard prediksi PM2.5 Kota Kendari (XGBoost) - Streamlit."""
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

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    html('<p class="side-title">Kondisi awal</p>'
         '<p class="side-note">Isi tanggal yang ingin diprediksi dan konsentrasi PM2.5 '
         'beberapa hari sebelumnya. Nilai ini dipakai di tab Prediksi dan Simulasi cuaca.</p>')
    target_date = st.date_input("Tanggal prediksi", value=date.today() + timedelta(days=1))
    is_holiday = st.checkbox("Hari libur nasional", value=False)

    d = hist_defaults or {"lag1": 11.9, "lag3": 11.9, "lag7": 11.9, "roll7": 11.9}
    st.markdown("**PM2.5 sebelumnya (µg/m³)**")
    lag1 = st.number_input("1 hari sebelum tanggal prediksi", 0.0, 500.0, float(d["lag1"]), 0.1)
    lag3 = st.number_input("3 hari sebelumnya", 0.0, 500.0, float(d["lag3"]), 0.1)
    lag7 = st.number_input("7 hari sebelumnya", 0.0, 500.0, float(d["lag7"]), 0.1)
    roll7 = st.number_input("Rata-rata 7 hari terakhir", 0.0, 500.0, float(d["roll7"]), 0.1)
    extra_vals = {}
    extras = extra_features(FEATURES)
    if extras:
        with st.expander("Data pendukung hari sebelumnya"):
            st.caption("Dipakai model versi terbaru. Nilai awal dari data terakhir atau rata-rata data latih.")
            for name in extras:
                extra_vals[name] = st.number_input(
                    FEATURE_LABELS.get(name, name), 0.0, 100000.0,
                    float(extra_default(name, FEATURES, scaler, history)), 0.1, key=f"extra_{name}")
    if hist_defaults:
        st.caption("Nilai awal diambil dari data terakhir di dataset.")
    else:
        st.caption("Nilai awal memakai rata-rata data latih. Ganti dengan pengukuran terbaru.")

# ------------------------------------------------------------------ hero
spectrum = "".join(f'<span style="background:{c.color}"></span>' for c in CATEGORIES)
trained = meta.get("created_at", "-")
html(f"""
<div class="hero">
  <h1>Perkiraan kualitas udara Kota Kendari</h1>
  <p>Model XGBoost memperkirakan konsentrasi PM2.5 harian dari data cuaca dan riwayat polusi,
  lalu mengubahnya menjadi indeks AQI US EPA 2024 beserta anjuran kesehatannya.</p>
  <div class="spectrum">{spectrum}</div>
  <div class="spectrum-labels"><span>Baik</span><span>Berbahaya</span></div>
  <div class="chips"><span class="chip">{meta.get('algorithm', 'XGBoost')}</span>
  <span class="chip">{len(FEATURES)} fitur</span><span class="chip">Dilatih {trained[:10]}</span></div>
</div>
""")

tab_pred, tab_sim, tab_perf, tab_hist, tab_about = st.tabs(
    ["Prediksi", "Simulasi cuaca", "Performa model", "Data historis", "Tentang"])


def run(weather: dict) -> float:
    row = build_features(target_date, **weather, lag1=lag1, lag3=lag3, lag7=lag7,
                         roll7=roll7, is_holiday=is_holiday, extra=extra_vals)
    return predict_pm25(model, scaler, FEATURES, [row])[0]


def scale_html(aqi: int) -> str:
    segs = [(c, c.i_high - c.i_low + (1 if c.i_low else 0)) for c in CATEGORIES[:5]]
    bar = "".join(f'<span style="background:{c.color};width:{w / 300 * 100:.2f}%"></span>' for c, w in segs)
    pos = min(aqi, 300) / 300 * 100
    return (f'<div class="scale"><div class="scale-bar">{bar}</div>'
            f'<div class="scale-pin" style="left:{pos:.1f}%"></div>'
            '<div class="scale-ticks"><span>0</span><span>50</span><span>100</span>'
            '<span>150</span><span>200</span><span>300+</span></div></div>')


# ================================================================== TAB 1
with tab_pred:
    section("Prediksi untuk satu hari", f"Tanggal prediksi: {target_date.strftime('%d %B %Y')}")
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        with st.container(border=True):
            html('<div class="panel" style="border:0;padding:0"><h3>Kondisi cuaca</h3>'
                 '<div class="sub">Gunakan prakiraan cuaca untuk tanggal tersebut.</div></div>')
            temp = st.slider("Suhu udara (°C)", 20.0, 40.0, 27.0, 0.1)
            hum = st.slider("Kelembapan (%)", 40.0, 100.0, 86.0, 0.5)
            rain = st.slider("Curah hujan (mm)", 0.0, 100.0, 5.0, 0.5)
            wind = st.slider("Kecepatan angin (km/jam)", 0.0, 30.0, 5.4, 0.1)

    weather = dict(temperature_c=temp, humidity_pct=hum, rainfall_mm=rain, wind_speed_kmh=wind)
    pm = run(weather)
    aqi = pm25_to_aqi(pm)
    cat = get_category(aqi)
    diff = (pm - WHO_DAILY_LIMIT) / WHO_DAILY_LIMIT * 100
    who_txt = (f"{abs(diff):.0f}% di atas batas WHO" if diff > 0 else f"{abs(diff):.0f}% di bawah batas WHO")

    with right:
        html(f"""
        <div class="result">
          <div class="result-band" style="background:{cat.color}">
            <div class="cat">{cat.name}</div>
            <div class="aqi">{aqi}</div>
            <div class="unit">Indeks AQI US EPA 2024</div>
          </div>
          <div class="result-body">
            <p>{cat.advice}</p>
            {scale_html(aqi)}
            <div class="facts" style="margin-top:1.1rem">
              <div class="fact"><b>{pm:.1f} µg/m³</b><span>PM2.5 diprediksi</span></div>
              <div class="fact"><b>{who_txt.split(' ')[0]}</b><span>{' '.join(who_txt.split(' ')[1:])} ({WHO_DAILY_LIMIT:.0f} µg/m³)</span></div>
            </div>
          </div>
        </div>
        """)

# ================================================================== TAB 2
with tab_sim:
    section("Bagaimana jika cuacanya berubah?",
            "Empat skenario dihitung dengan kondisi awal dari sidebar; hanya cuaca yang berbeda.")
    scenarios = {
        "Kemarau ekstrem": ("Skenario What-If notebook: tanpa hujan, angin tenang, suhu 33,5 °C.",
                            dict(temperature_c=33.5, humidity_pct=85.0, rainfall_mm=0.0, wind_speed_kmh=2.0)),
        "Cuaca normal": ("Mendekati rata-rata data latih.",
                         dict(temperature_c=27.0, humidity_pct=86.0, rainfall_mm=5.0, wind_speed_kmh=5.4)),
        "Angin kencang": ("Angin menyebarkan polutan lebih cepat.",
                          dict(temperature_c=27.0, humidity_pct=82.0, rainfall_mm=1.0, wind_speed_kmh=10.0)),
        "Hujan lebat": ("Hujan membersihkan partikel dari udara.",
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
          <div class="small">AQI {a} · {c.name}</div>
          <div class="small" style="margin-top:.5rem">Hujan {w['rainfall_mm']:.0f} mm · angin {w['wind_speed_kmh']:.1f} km/jam</div>
        </div>"""
    html(f'<div class="scen-grid">{cards}</div>')

    spread = max(preds) - min(preds)
    html(f"""<div class="note"><b>Selisih antar skenario: {spread:.2f} µg/m³.</b>
    Jika selisihnya kecil, model lebih banyak bergantung pada riwayat PM2.5 dan musim
    daripada cuaca hari itu. Ubah nilai PM2.5 di sidebar untuk melihat efeknya.</div>""")

# ================================================================== TAB 3
with tab_perf:
    m = meta["metrics"]
    section("Seberapa akurat modelnya?", "Hasil evaluasi pada data uji (20% data terakhir, urut waktu).")
    html(f"""
    <div class="metrics">
      <div class="metric"><div class="label">MAE</div><div class="value">{m['mae']:.2f}</div>
        <div class="hint">Rata-rata selisih absolut prediksi dan aktual, dalam µg/m³.</div></div>
      <div class="metric"><div class="label">RMSE</div><div class="value">{m['rmse']:.2f}</div>
        <div class="hint">Seperti MAE, tetapi lebih menghukum kesalahan besar.</div></div>
      <div class="metric"><div class="label">R²</div><div class="value">{m['r2']:.2f}</div>
        <div class="hint">Model menjelaskan sekitar {m['r2'] * 100:.0f}% variasi PM2.5 pada data uji.</div></div>
    </div>
    """)
    if m["r2"] < 0.6:
        html(f"""<div class="note warn"><b>Baca hasil ini dengan hati-hati.</b>
        R² {m['r2']:.2f} berarti sebagian besar variasi harian belum tertangkap model.
        Prediksi cocok sebagai gambaran kasar, bukan sebagai dasar keputusan tunggal.</div>""")

    bl = meta.get("baseline")
    if bl:
        better = m["r2"] > bl["persistence_r2"]
        html(f"""<div class="note{'' if better else ' warn'}"><b>Pembanding sederhana.</b>
        Menebak "sama seperti kemarin" menghasilkan MAE {bl['persistence_mae']:.2f} dan R² {bl['persistence_r2']:.2f}.
        Model XGBoost {'mengungguli' if better else 'belum mengungguli'} tebakan tersebut.</div>""")
    if not LEAK_FIXED:
        html("""<div class="note warn"><b>Model versi lama.</b> Fitur rata-rata 7 hari pada pelatihan memuat
        PM2.5 hari target (kebocoran data), jadi metrik di atas kemungkinan terlalu optimistis.
        Latih ulang dengan notebook versi perbaikan, lalu ganti isi folder <code>models/</code>.</div>""")

    section("Fitur yang paling berpengaruh")
    imp = feature_importance(model, FEATURES)
    if imp:
        top = imp[:10]
        mx = top[0][1] or 1
        bars = "".join(
            f'<div class="bar-row"><span>{FEATURE_LABELS.get(f, f)}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{v / mx * 100:.1f}%"></div></div>'
            f'<span class="bar-val">{v * 100:.1f}%</span></div>' for f, v in top)
        html(f'<div class="panel"><div class="sub">Skor kepentingan fitur dari model (10 teratas).</div>'
             f'<div class="bars">{bars}</div></div>')
    else:
        items = "".join(f"<li>{FEATURE_LABELS.get(f, f)}</li>" for f in meta.get("top_features", []))
        html(f'<div class="panel"><div class="sub">Lima fitur teratas dari metadata model.</div><ol>{items}</ol></div>')

# ================================================================== TAB 4
with tab_hist:
    section("Riwayat PM2.5", "Dari dataset hasil pipeline ETL Airflow.")
    if history is None:
        html("""<div class="note">Dataset belum ditemukan. Letakkan
        <code>kendari_aqi_pm25_dataset.csv</code> (kolom minimal <code>date</code> dan
        <code>pm25</code>) di folder <code>data/</code> repo Anda, lalu deploy ulang.</div>""")
    else:
        s = history["pm25"].dropna()
        over = (s > WHO_DAILY_LIMIT).mean() * 100
        html(f"""
        <div class="metrics">
          <div class="metric"><div class="label">Jumlah hari</div><div class="value">{len(s):,}</div>
            <div class="hint">{history['date'].min():%d %b %Y} – {history['date'].max():%d %b %Y}</div></div>
          <div class="metric"><div class="label">Rata-rata PM2.5</div><div class="value">{s.mean():.1f}</div>
            <div class="hint">µg/m³</div></div>
          <div class="metric"><div class="label">Hari di atas batas WHO</div><div class="value">{over:.0f}%</div>
            <div class="hint">Batas harian {WHO_DAILY_LIMIT:.0f} µg/m³</div></div>
        </div>""")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history["date"], y=history["pm25"], mode="lines",
                                 line=dict(color="#1f6f78", width=1.6), name="PM2.5"))
        fig.add_hline(y=WHO_DAILY_LIMIT, line_dash="dash", line_color="#d64545",
                      annotation_text="Batas WHO", annotation_position="top left")
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="#ffffff", font=dict(family="Source Sans 3", color="#12303a"),
                          yaxis_title="µg/m³", showlegend=False,
                          xaxis=dict(gridcolor="#eef2f1"), yaxis=dict(gridcolor="#eef2f1"))
        st.plotly_chart(fig, use_container_width=True)

        monthly = history.assign(bulan=history["date"].dt.month).groupby("bulan")["pm25"].mean()
        names = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
        fig2 = go.Figure(go.Bar(x=[names[i - 1] for i in monthly.index], y=monthly.values,
                                marker_color="#1f6f78"))
        fig2.add_hline(y=WHO_DAILY_LIMIT, line_dash="dash", line_color="#d64545")
        fig2.update_layout(title="Rata-rata PM2.5 per bulan", height=300, margin=dict(l=0, r=0, t=40, b=0),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
                           font=dict(family="Source Sans 3", color="#12303a"), yaxis_title="µg/m³",
                           yaxis=dict(gridcolor="#eef2f1"))
        st.plotly_chart(fig2, use_container_width=True)

# ================================================================== TAB 5
with tab_about:
    section("Tentang dashboard ini")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        html("""<div class="panel"><h3>Alur data</h3>
        <div class="sub">Dari sumber terbuka sampai prediksi.</div>
        <p><b>1. Ekstraksi</b> – Open-Meteo (polutan), NASA POWER (cuaca), Nager.Date (libur) lewat DAG Airflow.</p>
        <p><b>2. Transformasi</b> – gabung per tanggal, buat fitur lag, rata-rata bergulir, dan encoding siklik.</p>
        <p><b>3. Model</b> – XGBoost Regressor, split kronologis 80/20 tanpa shuffle.</p>
        <p><b>4. Konversi</b> – PM2.5 ke AQI US EPA 2024 dengan interpolasi linear antar-breakpoint.</p></div>""")
    with c2:
        html(f"""<div class="panel"><h3>Batasan</h3>
        <div class="sub">Hal yang perlu diketahui pengguna.</div>
        <p>Prediksi bersifat perkiraan dan bukan pengganti alat ukur resmi.</p>
        <p>Akurasi bergantung pada kebenaran nilai PM2.5 sebelumnya yang Anda masukkan.</p>
        {LEAK_NOTE}
        <p>Model dilatih dari data satelit/reanalisis, bukan sensor di lapangan, sehingga bisa berbeda dari kondisi di tiap titik kota.</p></div>""")
    section("Daftar fitur model")
    chips = "".join(f'<span class="chip">{FEATURE_LABELS.get(f, f)}</span>' for f in FEATURES)
    html(f'<div class="chips" style="margin-top:0">{chips}</div>')
