"""Mesin analitik deterministik OASIS.

Semua perhitungan dilakukan di sini atas data yang sudah divalidasi. Fungsi
mengembalikan angka beserta metode dan cakupan observasi agar dapat diaudit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from . import stats_lite as sstats  # pengganti scipy.stats (tanpa DLL)

CONFIRMED = {"terkonfirmasi_kamus", "label_sumber", "dikonfirmasi_pengguna"}


# ---------------------------------------------------------------------------
# Format angka (gaya Indonesia)
# ---------------------------------------------------------------------------
def fmt_num(x: float | None, dec: int | None = None) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    ax = abs(x)
    if dec is None:
        dec = 2 if ax < 1000 else (1 if ax < 100000 else 0)
    s = f"{x:,.{dec}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def unit_text(info: dict) -> str:
    u = info.get("unit")
    if info.get("unit_status") == "belum_terkonfirmasi" or not isinstance(u, str) or not u:
        return "satuan belum terkonfirmasi"
    return str(info["unit"])


def fmt_val(x: float | None, info: dict) -> str:
    u = unit_text(info)
    if u == "%":
        return f"{fmt_num(x)}%"
    if u == "satuan belum terkonfirmasi":
        return f"{fmt_num(x)} (satuan belum terkonfirmasi)"
    return f"{fmt_num(x)} {u}"


def period_label(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return "–"
    bulan = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    return f"{bulan[ts.month - 1]} {ts.year}"


# ---------------------------------------------------------------------------
# Jenis perubahan
# ---------------------------------------------------------------------------
def change_mode(info: dict, s: pd.Series | None = None) -> str:
    """'pp' untuk rasio %, 'pct' untuk besaran positif (stok/arus/indeks/jumlah), 'abs' bila belum pasti."""
    if info.get("measure_type") == "rasio" and unit_text(info) == "%":
        return "pp"
    if info.get("measure_type") in ("stok", "arus", "indeks", "jumlah"):
        if s is None or (s.dropna() > 0).all():
            return "pct"
        return "abs"
    return "abs"


def change_label(mode: str, info: dict) -> str:
    return {"pp": "pp (percentage points)", "pct": "%",
            "abs": f"selisih ({unit_text(info)})"}[mode]


def monthly_changes(s: pd.Series, info: dict) -> pd.DataFrame:
    """Hitung MoM & YoY dengan memastikan pembanding adalah bulan kalender yang tepat (bukan baris sebelumnya)."""
    s = s.dropna().sort_index()
    if s.empty:
        return pd.DataFrame(columns=["nilai", "mom", "yoy", "mom_abs", "yoy_abs"])
    idx = pd.date_range(s.index.min(), s.index.max(), freq="ME")
    full = s.reindex(idx)
    prev = full.shift(1)
    prev12 = full.shift(12)
    mode = change_mode(info, s)
    df = pd.DataFrame({"nilai": full, "mom_abs": full - prev, "yoy_abs": full - prev12})
    if mode == "pct":
        df["mom"] = (full / prev - 1) * 100
        df["yoy"] = (full / prev12 - 1) * 100
    else:
        df["mom"] = df["mom_abs"]
        df["yoy"] = df["yoy_abs"]
    df = df.loc[s.index]
    df.attrs["mode"] = mode
    return df


# ---------------------------------------------------------------------------
# Statistik deskriptif & tren
# ---------------------------------------------------------------------------
def describe(s: pd.Series) -> dict:
    s = s.dropna()
    n = len(s)
    if n == 0:
        return {"n": 0}
    out = {
        "n": n, "awal": s.index.min(), "akhir": s.index.max(),
        "terakhir": float(s.iloc[-1]), "pertama": float(s.iloc[0]),
        "maks": float(s.max()), "periode_maks": s.idxmax(),
        "min": float(s.min()), "periode_min": s.idxmin(),
        "rata_rata": float(s.mean()), "median": float(s.median()),
        "std": float(s.std(ddof=1)) if n > 1 else np.nan,
        "q1": float(s.quantile(0.25)), "q3": float(s.quantile(0.75)),
    }
    out["iqr"] = out["q3"] - out["q1"]
    out["n_maks"] = int((s == out["maks"]).sum())
    out["n_min"] = int((s == out["min"]).sum())
    return out


def trend(s: pd.Series, info: dict) -> dict:
    """Tren linier sederhana (OLS terhadap indeks bulan). Uji signifikansi bersifat indikatif."""
    s = s.dropna()
    if len(s) < 6:
        return {"ok": False, "alasan": f"Observasi terlalu sedikit ({len(s)}) untuk tren (minimal 6)."}
    t = np.array([(p.year - s.index[0].year) * 12 + (p.month - s.index[0].month) for p in s.index], dtype=float)
    res = sstats.linregress(t, s.values)
    slope_m = float(res.slope)
    mode = change_mode(info, s)
    mean = float(s.mean())
    per_year = slope_m * 12
    if res.pvalue >= 0.05:
        arah = "relatif datar (kemiringan tidak signifikan secara statistik, p ≥ 0,05)"
    else:
        arah = "naik" if slope_m > 0 else "turun"
    if mode == "pp":
        laju = f"{fmt_num(per_year)} pp per tahun"
    elif mode == "pct" and mean != 0:
        laju = f"{fmt_num(per_year / mean * 100)}% per tahun (relatif terhadap rata-rata)"
    else:
        laju = f"{fmt_num(per_year)} {unit_text(info)} per tahun"
    return {"ok": True, "slope_bulan": slope_m, "slope_tahun": per_year, "r2": float(res.rvalue ** 2),
            "p": float(res.pvalue), "arah": arah, "laju": laju, "n": len(s),
            "metode": "Regresi linier OLS nilai terhadap waktu (bulan); p-value indikatif karena autokorelasi deret "
                      "waktu tidak dikoreksi."}


def sum_allowed(info: dict, flags: list[dict]) -> tuple[bool, str]:
    kinds = {f["jenis"] for f in flags}
    mt = info.get("measure_type")
    if mt == "stok" or mt == "jumlah":
        return False, ("Indikator ini adalah STOK/posisi akhir periode; menjumlahkan nilai antarbulan tidak bermakna. "
                       "Gunakan nilai akhir periode, rata-rata, atau perubahan.")
    if mt in ("rasio", "indeks"):
        return False, "Rasio/indeks tidak dapat dijumlahkan antarbulan. Gunakan rata-rata, median, atau nilai akhir."
    if mt == "arus" and "indikasi_ytd" in kinds:
        return False, ("Indikator terindikasi KUMULATIF tahun berjalan (reset Januari). Penjumlahan antarbulan akan "
                       "menghitung ganda. Gunakan nilai Desember (posisi akhir tahun) atau YoY bulan yang sama.")
    if mt == "arus" and info.get("measure_certain"):
        return True, "Indikator arus 'selama periode' — penjumlahan antarbulan valid."
    return False, ("Klasifikasi stok/arus belum pasti menurut kamus; penjumlahan antarbulan tidak dilakukan. "
                   "Gunakan rata-rata atau nilai akhir periode.")


# ---------------------------------------------------------------------------
# Penyelarasan & transformasi perbandingan
# ---------------------------------------------------------------------------
def align(series: dict[str, pd.Series]) -> dict:
    """Sejajarkan periode beririsan. Kembalikan tabel irisan, gabungan, dan periode hilang per seri."""
    clean = {k: v.dropna() for k, v in series.items()}
    union = sorted(set().union(*[set(v.index) for v in clean.values()])) if clean else []
    inter = sorted(set.intersection(*[set(v.index) for v in clean.values()])) if clean else []
    union_df = pd.DataFrame(index=pd.DatetimeIndex(union, name="periode"))
    for k, v in clean.items():
        union_df[k] = v
    missing = {k: [p for p in union if p not in set(v.index)] for k, v in clean.items()}
    return {"irisan": union_df.loc[inter] if inter else union_df.iloc[0:0], "gabungan": union_df,
            "hilang": missing, "n_irisan": len(inter),
            "rentang_irisan": (inter[0], inter[-1]) if inter else (None, None)}


def index_base100(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    if df.empty:
        return df, None
    base = df.index[0]
    return df.divide(df.iloc[0]).multiply(100), base


def correlation(a: pd.Series, b: pd.Series, info_a: dict, info_b: dict, flags_a=(), flags_b=()) -> dict:
    """Korelasi eksploratif pada level dan pada perubahan bulanan (lebih tahan tren bersama)."""
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    out = {"n_level": len(df)}
    if len(df) >= 6 and df["a"].nunique() > 1 and df["b"].nunique() > 1:
        r, p = sstats.pearsonr(df["a"], df["b"])
        rho, ps = sstats.spearmanr(df["a"], df["b"])
        out.update({"r_level": float(r), "p_level": float(p), "rho_level": float(rho)})

    def chg(s, info, flags):
        if any(f["jenis"] == "indikasi_ytd" for f in flags):
            c = monthly_changes(s, info)["yoy"]
            return c, "YoY (seri terindikasi kumulatif)"
        c = monthly_changes(s, info)["mom"]
        return c, "MoM"

    ca, la = chg(a, info_a, flags_a)
    cb, lb = chg(b, info_b, flags_b)
    dc = pd.concat([ca.rename("a"), cb.rename("b")], axis=1).dropna()
    out["n_perubahan"] = len(dc)
    out["basis_perubahan"] = f"{la} vs {lb}"
    if len(dc) >= 6 and dc["a"].nunique() > 1 and dc["b"].nunique() > 1:
        r, p = sstats.pearsonr(dc["a"], dc["b"])
        out.update({"r_perubahan": float(r), "p_perubahan": float(p)})
    out["catatan"] = ("Korelasi bersifat EKSPLORATIF: tidak menunjukkan sebab-akibat maupun transmisi risiko. Korelasi "
                      "level sering semu bila kedua seri sama-sama bertren; perhatikan korelasi perubahan.")
    return out


def strength(r: float | None) -> str:
    if r is None or not np.isfinite(r):
        return "tidak dapat dihitung"
    a = abs(r)
    lvl = "sangat lemah" if a < 0.2 else "lemah" if a < 0.4 else "sedang" if a < 0.6 else "kuat" if a < 0.8 else "sangat kuat"
    return f"{lvl} {'positif' if r > 0 else 'negatif'}"
