"""Konversi PM2.5 (µg/m³) ke US EPA AQI (revisi 2024)."""
from dataclasses import dataclass

WHO_DAILY_LIMIT = 15.0  # µg/m³, pedoman WHO 2021 (rata-rata 24 jam)


@dataclass(frozen=True)
class Category:
    name: str
    color: str
    c_low: float
    c_high: float
    i_low: int
    i_high: int
    advice: str


CATEGORIES = [
    Category("Baik", "#2e9e5b", 0.0, 9.0, 0, 50,
             "Udara bersih. Aktivitas di luar ruangan aman untuk semua orang."),
    Category("Sedang", "#e0b100", 9.1, 35.4, 51, 100,
             "Kualitas udara dapat diterima. Orang yang sangat sensitif sebaiknya mengurangi aktivitas berat di luar."),
    Category("Tidak sehat bagi kelompok sensitif", "#f08a24", 35.5, 55.4, 101, 150,
             "Anak-anak, lansia, dan penderita asma atau penyakit jantung sebaiknya membatasi aktivitas di luar ruangan."),
    Category("Tidak sehat", "#d64545", 55.5, 125.4, 151, 200,
             "Semua orang mulai terdampak. Kurangi aktivitas berat di luar dan gunakan masker bila harus keluar."),
    Category("Sangat tidak sehat", "#8a4fa3", 125.5, 225.4, 201, 300,
             "Peringatan kesehatan. Hindari aktivitas di luar ruangan dan tutup ventilasi rumah."),
    Category("Berbahaya", "#7a2237", 225.5, 325.4, 301, 500,
             "Kondisi darurat. Tetap di dalam ruangan dan ikuti arahan otoritas kesehatan."),
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
