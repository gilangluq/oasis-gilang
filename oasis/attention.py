"""Deteksi anomali, pemeriksaan threshold, dan kotak 'Perlu Perhatian'.

Metode anomali: skor-z robust (median & MAD) atas perubahan bulanan. Untuk seri
yang terindikasi kumulatif tahun berjalan, perubahan dihitung atas nilai hasil
de-kumulasi agar reset Januari tidak dianggap anomali.

Threshold berasal dari ``config/thresholds.json`` atau input pengguna dan selalu
diberi label sumbernya (default: threshold simulasi) — BUKAN batas regulasi.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analytics import change_label, change_mode, fmt_num, fmt_val, monthly_changes, period_label
from .catalog import load_json
from .data_prep import ytd_to_monthly

Z_THRESHOLD = 3.5
MIN_CHANGES = 12
PRIORITY_ORDER = {"Tinggi": 0, "Sedang": 1, "Rendah": 2}


@dataclass
class Threshold:
    nilai: float
    arah: str  # 'atas' | 'bawah'
    label: str
    sumber: str


def load_thresholds() -> list[dict]:
    try:
        return load_json("thresholds.json").get("threshold", [])
    except FileNotFoundError:  # pragma: no cover
        return []


def thresholds_for(info: dict, overrides: dict[str, list[dict]] | None = None) -> list[Threshold]:
    overrides = overrides or {}
    if info["indicator_id"] in overrides:
        return [Threshold(**t) for t in overrides[info["indicator_id"]]]
    out = []
    for t in load_thresholds():
        if t.get("konsep") == info.get("concept_id") or t.get("indicator_id") == info.get("indicator_id"):
            out.append(Threshold(nilai=float(t["nilai"]), arah=t["arah"], label=t["label"], sumber=t["sumber"]))
    return out


def detect_anomalies(s: pd.Series, info: dict, flags: list[dict], z_thr: float = Z_THRESHOLD) -> tuple[pd.DataFrame, str]:
    """Kembalikan (tabel anomali, deskripsi metode)."""
    s = s.dropna().sort_index()
    kinds = {f["jenis"] for f in flags}
    cols = ["periode", "nilai", "perubahan", "skor_z", "alasan", "metode"]
    if "konstan" in kinds or len(s) < MIN_CHANGES + 1:
        return pd.DataFrame(columns=cols), (f"Deteksi anomali tidak dijalankan: observasi < {MIN_CHANGES + 1} atau seri "
                                            f"konstan.")
    if "indikasi_ytd" in kinds:
        base = ytd_to_monthly(s)
        idx = pd.date_range(base.index.min(), base.index.max(), freq="ME")
        b = base.reindex(idx)
        chg = (b - b.shift(1)).loc[s.index]
        mode_txt = ("perubahan nilai bulanan hasil de-kumulasi heuristik (seri terindikasi kumulatif tahun "
                    "berjalan), dalam satuan sumber")
        unit_chg = ""
    else:
        ch = monthly_changes(s, info)
        chg = ch["mom"]
        mode = ch.attrs.get("mode", change_mode(info, s))
        mode_txt = f"perubahan MoM ({change_label(mode, info)})"
        unit_chg = {"pp": "pp", "pct": "%", "abs": ""}[mode]
    c = chg.dropna()
    if len(c) < MIN_CHANGES:
        return pd.DataFrame(columns=cols), f"Deteksi anomali tidak dijalankan: perubahan valid < {MIN_CHANGES}."
    med = float(c.median())
    mad = float((c - med).abs().median()) * 1.4826
    scale_txt = "MAD×1,4826"
    if mad == 0:
        mad = float(c.std(ddof=1))
        scale_txt = "simpangan baku (MAD = 0)"
    if not mad or not np.isfinite(mad):
        return pd.DataFrame(columns=cols), "Deteksi anomali tidak dijalankan: variasi perubahan nol."
    z = (c - med) / mad
    method = (f"Skor-z robust atas {mode_txt}: z = (perubahan − median) / {scale_txt}; anomali bila |z| ≥ "
              f"{fmt_num(z_thr, 1)}. Dihitung atas {len(c)} perubahan pada periode terpilih "
              f"({period_label(c.index.min())}–{period_label(c.index.max())}).")
    rows = []
    for p, zv in z[z.abs() >= z_thr].items():
        v = float(s.loc[p])
        cv = float(c.loc[p])
        rows.append({
            "periode": p, "nilai": v, "perubahan": cv, "skor_z": float(zv),
            "alasan": " ".join((f"Perubahan {'+' if cv >= 0 else ''}{fmt_num(cv)} {unit_chg} pada {period_label(p)} "
                                 f"jauh dari pola biasa (median {fmt_num(med)} {unit_chg}, skala {fmt_num(mad)}); "
                                 f"skor-z robust {fmt_num(float(zv), 1)}.").split()).replace(" ,", ","),
            "metode": method,
        })
    return pd.DataFrame(rows, columns=cols), method


def check_thresholds(s: pd.Series, info: dict, ths: list[Threshold]) -> list[dict]:
    s = s.dropna()
    out = []
    for t in ths:
        mask = s > t.nilai if t.arah == "atas" else s < t.nilai
        breached = s[mask]
        out.append({"threshold": t, "n_lewat": int(mask.sum()), "periode_lewat": list(breached.index),
                    "terakhir_lewat": bool(len(s) and mask.iloc[-1])})
    return out


ACTIONS = {
    "threshold": ("Verifikasi angka dengan laporan sumber periode tersebut; telusuri kontributor utama dan bandingkan "
                  "dengan periode yang sama tahun lalu. Pastikan dasar threshold sebelum menyimpulkan pelanggaran."),
    "anomali": ("Periksa kemungkinan perubahan metodologi/pelaporan, koreksi data, merger/perubahan cakupan, atau "
                "peristiwa khusus pada periode tersebut; konfirmasi ke pemilik data."),
    "memburuk": "Pantau rilis berikutnya dan dalami komponen pembentuk indikator untuk memastikan arah pergerakan.",
    "kualitas": "Konfirmasi satuan/definisi/pemetaan field dengan pemilik data sebelum menarik kesimpulan analitis.",
}


def attention_items(s: pd.Series, info: dict, flags: list[dict], anomalies: pd.DataFrame,
                    threshold_hits: list[dict]) -> list[dict]:
    s = s.dropna().sort_index()
    items = []
    if s.empty:
        return items
    base = {"indikator": info["label_tampil"], "subsektor": info["sheet"], "indicator_id": info["indicator_id"]}
    last_p = s.index.max()
    for h in threshold_hits:
        t = h["threshold"]
        if h["n_lewat"] == 0:
            continue
        lvl = "Tinggi" if h["terakhir_lewat"] else "Sedang"
        per = period_label(last_p) if h["terakhir_lewat"] else period_label(h["periode_lewat"][-1])
        items.append({**base, "periode": per, "prioritas": lvl, "jenis": "threshold",
                      "alasan": f"Nilai {'di atas' if t.arah == 'atas' else 'di bawah'} {t.label} pada "
                                f"{h['n_lewat']} periode{' termasuk periode terakhir' if h['terakhir_lewat'] else ''}.",
                      "angka": f"Terakhir {fmt_val(float(s.iloc[-1]), info)}; batas {fmt_num(t.nilai)} ({t.sumber}).",
                      "tindakan": ACTIONS["threshold"]})
    if not anomalies.empty:
        recent_cut = s.index[-3] if len(s) >= 3 else s.index[0]
        for _, a in anomalies.iterrows():
            recent = a["periode"] >= recent_cut
            big = abs(a["skor_z"]) >= 5
            lvl = "Tinggi" if (recent and big) else ("Sedang" if (recent or big) else "Rendah")
            items.append({**base, "periode": period_label(a["periode"]), "prioritas": lvl, "jenis": "anomali",
                          "alasan": a["alasan"], "angka": f"Nilai {fmt_val(a['nilai'], info)}; z = {fmt_num(a['skor_z'], 1)}.",
                          "tindakan": ACTIONS["anomali"]})
    arah = info.get("arah_perhatian")
    arah = arah if isinstance(arah, str) else None
    if arah and len(s) >= 4 and not any(f["jenis"] == "indikasi_ytd" for f in flags):
        d = s.diff().iloc[-3:]
        worse = (d > 0).all() if arah == "naik_memburuk" else (d < 0).all()
        if worse:
            ch = monthly_changes(s, info)
            mode = ch.attrs.get("mode")
            tot = float(s.iloc[-1] - s.iloc[-4])
            items.append({**base, "periode": f"{period_label(s.index[-4])}–{period_label(last_p)}", "prioritas": "Sedang",
                          "jenis": "memburuk",
                          "alasan": f"Bergerak ke arah {'naik' if arah == 'naik_memburuk' else 'turun'} 3 bulan berturut-turut "
                                    f"(arah perhatian dikonfigurasi: {arah.replace('_', ' ')}).",
                          "angka": f"Perubahan kumulatif 3 bulan: {'+' if tot >= 0 else ''}{fmt_num(tot)} "
                                   f"{'pp' if mode == 'pp' else ('' if info.get('unit_status') == 'belum_terkonfirmasi' else info.get('unit') or '')}.",
                          "tindakan": ACTIONS["memburuk"]})
    q_kinds = {"aturan_gagal": "Uji aturan eksplisit gagal", "identik_lintas_sheet": "Seri identik dengan sheet lain",
               "pemetaan_posisi": "Pemetaan field berdasarkan posisi", "konflik_satuan": "Konflik satuan kamus vs header"}
    for f in flags:
        if f["jenis"] in q_kinds:
            items.append({**base, "periode": "seluruh periode", "prioritas": "Rendah", "jenis": "kualitas",
                          "alasan": f"{q_kinds[f['jenis']]}: {f['teks']}", "angka": "–", "tindakan": ACTIONS["kualitas"]})
    return items


def sort_items(items: list[dict]) -> list[dict]:
    return sorted(items, key=lambda x: (PRIORITY_ORDER.get(x["prioritas"], 9), x["jenis"] != "threshold"))
