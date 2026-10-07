"""Konversi PM2.5 (µg/m³) ke US EPA AQI (revisi 2024)."""
from dataclasses import dataclass

WHO_DAILY_LIMIT = 15.0  # µg/m³, pedoman WHO 2021 (rata-rata 24 jam)


@dataclass(frozen=True)
class Category:
    level: int
    name: str
    color: str
    c_low: float
    c_high: float
    i_low: int
    i_high: int
    advice: str


CATEGORIES = [
    Category(1, "Baik", "#2e9e5b", 0.0, 9.0, 0, 50,
             "Udara bersih. Kualitas udara sangat baik dan aktivitas luar ruangan aman bagi seluruh kalangan."),
    Category(2, "Sedang", "#e0b100", 9.1, 35.4, 51, 100,
             "Kualitas udara dapat diterima. Individu yang luar biasa sensitif disarankan mengurangi aktivitas fisik berat di luar."),
    Category(3, "Tidak sehat bagi kelompok sensitif", "#f08a24", 35.5, 55.4, 101, 150,
             "Anak-anak, lansia, dan penderita asma/jantung sebaiknya membatasi aktivitas berat di luar ruangan."),
    Category(4, "Tidak sehat", "#d64545", 55.5, 125.4, 151, 200,
             "Masyarakat umum mulai merasakan dampak kesehatan. Gunakan masker medis/N95 bila harus beraktivitas di luar."),
    Category(5, "Sangat tidak sehat", "#8a4fa3", 125.5, 225.4, 201, 300,
             "Peringatan kesehatan darurat. Hindari segala aktivitas fisik di luar ruangan dan tutup ventilasi rumah."),
    Category(6, "Berbahaya", "#7a2237", 225.5, 325.4, 301, 500,
             "Tingkat bahaya darurat bagi seluruh populasi. Wajib berada di dalam ruangan dengan alat pemurni udara."),
]


def pm25_to_aqi(conc: float) -> int:
    """Interpolasi linear antar-breakpoint EPA. Konsentrasi dipotong ke 0,1 µg/m³."""
    c = max(0.0, int(conc * 10) / 10)
    for cat in CATEGORIES:
        if c <= cat.c_high:
            span_c = cat.c_high - cat.c_low
            span_i = cat.i_high - cat.i_low
            return round((span_i / span_c) * (c - cat.c_low) + cat.i_low)
    return 500


def get_category(aqi: int) -> Category:
    for cat in CATEGORIES:
        if aqi <= cat.i_high:
            return cat
    return CATEGORIES[-1]
