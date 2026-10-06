# Dashboard Prediksi PM2.5 Kota Kendari

Dashboard Streamlit untuk model XGBoost prediksi PM2.5 harian dan konversi ke AQI US EPA 2024.

## Struktur
```
app.py                  # halaman utama
utils/aqi.py            # konversi PM2.5 -> AQI
utils/predictor.py      # load model, bangun fitur, prediksi
assets/style.css        # HTML/CSS kustom
models/                 # xgboost_pm25_model.pkl, scaler.pkl, model_metadata.json
data/                   # (opsional) kendari_aqi_pm25_dataset.csv untuk tab Data historis
.streamlit/config.toml  # tema
requirements.txt
```

## Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud
1. Push folder ini ke repositori GitHub.
2. Buka https://share.streamlit.io, pilih **New app**, pilih repo dan branch.
3. Isi **Main file path** dengan `app.py`, lalu **Deploy**.

## Catatan penting
- `scikit-learn` dikunci ke 1.6.1 karena scaler dibuat dengan versi itu.
- Fitur waktu mengikuti notebook: `month_*` = bulan/12, `day_*` = hari dalam pekan/7.
- Di notebook, `pm25_rolling_mean_7d` dihitung dengan `rolling(7)` yang menyertakan hari target (kebocoran data).
  Di dashboard nilainya diisi dari 7 hari sebelumnya, jadi hasil bisa berbeda dari metrik evaluasi.

## Memperbarui model
Jalankan `kendari_pm25_xgboost_colab_fixed.ipynb` di Colab, lalu ganti 3 file di `models/`
(`xgboost_pm25_model.pkl`, `scaler.pkl`, `model_metadata.json`) dan `data/kendari_aqi_pm25_dataset.csv`.
Dashboard otomatis membaca metrik dan pembanding baseline dari metadata baru.
