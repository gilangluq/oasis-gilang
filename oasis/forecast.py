"""Forecasting sederhana & dapat dijelaskan.

Metode baseline: naive, seasonal naive (musim 12 bulan), drift (random walk +
drift), dan tren linier. Pemilihan metode memakai evaluasi rolling-origin
(backtest). Interval ketidakpastian = kuantil empiris galat backtest per
langkah horizon (~80%). Hasil adalah PERKIRAAN ilustratif, bukan prediksi resmi
OJK.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ALLOWED_H = (3, 6, 12)
METHOD_LABEL = {
    "naive": "Naive (nilai terakhir)",
    "snaive": "Seasonal naive (nilai bulan sama tahun lalu)",
    "drift": "Drift (rata-rata perubahan historis)",
    "tren": "Tren linier (OLS)",
}


@dataclass
class ForecastResult:
    ok: bool
    alasan: str = ""
    metode: str | None = None
    horizon: int | None = None
    n_latih: int = 0
    latih_awal: pd.Timestamp | None = None
    latih_akhir: pd.Timestamp | None = None
    prediksi: pd.DataFrame | None = None  # index periode; kolom: prediksi, bawah, atas
    evaluasi: pd.DataFrame | None = None  # per metode: MAE, RMSE, MAPE, n_origin
    catatan: list[str] = field(default_factory=list)

    @property
    def metode_label(self) -> str:
        return METHOD_LABEL.get(self.metode or "", "-")


def _fc(method: str, y: np.ndarray, h: int) -> np.ndarray:
    n = len(y)
    if method == "naive":
        return np.repeat(y[-1], h)
    if method == "snaive":
        return np.array([y[n - 12 + (i % 12)] for i in range(h)])
    if method == "drift":
        slope = (y[-1] - y[0]) / (n - 1)
        return y[-1] + slope * np.arange(1, h + 1)
    if method == "tren":
        t = np.arange(n)
        b, a = np.polyfit(t, y, 1)
        return a + b * np.arange(n, n + h)
    raise ValueError(method)


def min_obs(h: int) -> int:
    return max(12, 3 * h)


def forecast(s: pd.Series, h: int, flags: list[dict] | None = None, info: dict | None = None) -> ForecastResult:
    flags = flags or []
    kinds = {f["jenis"] for f in flags}
    if h not in ALLOWED_H:
        return ForecastResult(False, f"Horizon {h} tidak didukung; pilih 3, 6, atau 12 bulan.")
    s = s.sort_index()
    if s.dropna().empty:
        return ForecastResult(False, "Seri kosong; proyeksi tidak dibuat.")
    # gunakan bagian kontigu terakhir (tanpa nilai hilang/bulan hilang)
    s = s.dropna()
    full = pd.date_range(s.index.min(), s.index.max(), freq="ME")
    if len(full) != len(s):
        missing = full.difference(s.index)
        last_gap = missing.max()
        s = s[s.index > last_gap]
        note_gap = (f"Seri memiliki {len(missing)} bulan hilang; hanya segmen kontigu terakhir "
                    f"({s.index.min():%Y-%m}–{s.index.max():%Y-%m}) yang dipakai.")
    else:
        note_gap = None
    n = len(s)
    if "konstan" in kinds or s.nunique() <= 1:
        return ForecastResult(False, "Seri konstan; proyeksi tidak bermakna.")
    if n < min_obs(h):
        return ForecastResult(False, f"Observasi kontigu ({n}) kurang dari batas wajar {min_obs(h)} untuk horizon "
                                     f"{h} bulan (aturan: minimal 3× horizon dan ≥ 12). Pilih horizon lebih pendek.",
                              n_latih=n)
    y = s.values.astype(float)
    methods = ["naive", "drift", "tren"] + (["snaive"] if n >= 36 else [])
    min_train = 24 if "snaive" in methods else 12
    n_origins = min(12, n - h - min_train + 1)
    errors: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    rows = []
    if n_origins >= 3:
        for m in methods:
            for k in range(n_origins):
                o = n - h - k
                pred = _fc(m, y[:o], h)
                errors[m].append(y[o:o + h] - pred)
            e = np.vstack(errors[m])
            act = np.vstack([y[n - h - k:n - k] for k in range(n_origins)])
            mape = float(np.mean(np.abs(e / act)) * 100) if np.all(np.abs(act) > 1e-9) else np.nan
            rows.append({"metode": m, "label": METHOD_LABEL[m], "MAE": float(np.mean(np.abs(e))),
                         "RMSE": float(np.sqrt(np.mean(e ** 2))), "MAPE (%)": mape, "n_origin": n_origins})
        ev = pd.DataFrame(rows).sort_values(["MAE"]).reset_index(drop=True)
        best = ev.iloc[0]["metode"]
        sel_note = (f"Metode dipilih dengan backtest rolling-origin ({n_origins} titik asal, horizon {h}) — MAE "
                    f"terendah.")
    else:
        ev = None
        best = "snaive" if "snaive" in methods else "naive"
        sel_note = "Observasi tidak cukup untuk backtest; dipakai metode baseline default tanpa metrik kesalahan."
    pred = _fc(best, y, h)
    idx = pd.date_range(s.index.max() + pd.offsets.MonthEnd(1), periods=h, freq="ME")
    if ev is not None:
        e = np.vstack(errors[best])
        lo = np.quantile(e, 0.10, axis=0)
        hi = np.quantile(e, 0.90, axis=0)
        # pastikan melebar monoton terhadap horizon
        lo = np.minimum.accumulate(lo)
        hi = np.maximum.accumulate(hi)
        interval_note = "Interval perkiraan ~80% = kuantil 10%–90% galat backtest per langkah (empiris, bukan jaminan)."
    else:
        sd = np.std(np.diff(y), ddof=1)
        lo = -1.28 * sd * np.sqrt(np.arange(1, h + 1))
        hi = -lo
        interval_note = "Interval perkiraan ~80% = ±1,28 × simpangan baku perubahan bulanan × √langkah (aproksimasi)."
    lower, upper = pred + lo, pred + hi
    if np.all(y >= 0):
        lower = np.maximum(lower, 0)
        pred = np.maximum(pred, 0)
    out = pd.DataFrame({"prediksi": pred, "bawah": lower, "atas": upper}, index=idx)
    notes = [sel_note, interval_note,
             "Proyeksi adalah perkiraan statistik ilustratif atas pola historis — BUKAN prediksi resmi OJK."]
    if note_gap:
        notes.insert(0, note_gap)
    if "indikasi_ytd" in kinds:
        notes.append("Seri terindikasi kumulatif tahun berjalan; seasonal naive menangkap pola reset Januari bila "
                     "tersedia. Tafsirkan proyeksi lintas Januari dengan hati-hati.")
    if info and info.get("unit_status") == "belum_terkonfirmasi":
        notes.append("Satuan seri belum terkonfirmasi; angka proyeksi mengikuti satuan sumber.")
    return ForecastResult(True, "", best, h, n, s.index.min(), s.index.max(), out, ev, notes)
