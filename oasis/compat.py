"""Matriks kompatibilitas indikator.

Menentukan cara perbandingan yang SAH untuk sepasang indikator:

* ``absolut``     – nilai dapat diplot/dibandingkan langsung pada sumbu yang sama.
* ``perubahan``   – hanya perubahan (pp untuk rasio %) yang sebanding.
* ``indeks``      – gunakan indeks basis 100 / % perubahan (besaran berbeda atau satuan belum terkonfirmasi).
* ``berdampingan``– hanya visual berdampingan atau dua sumbu berlabel jelas.
"""
from __future__ import annotations

import pandas as pd

from .analytics import CONFIRMED, unit_text
from .catalog import find_conversion

LEVELS = {
    "absolut": {"skor": 3, "label": "Nilai absolut", "mode": ["sumbu_sama", "indeks_100", "perubahan", "dua_sumbu", "berdampingan"]},
    "perubahan": {"skor": 2, "label": "Perubahan (pp)", "mode": ["perubahan", "dua_sumbu", "berdampingan"]},
    "indeks": {"skor": 1, "label": "Indeks basis 100 / % perubahan", "mode": ["indeks_100", "perubahan", "dua_sumbu", "berdampingan"]},
    "berdampingan": {"skor": 0, "label": "Hanya berdampingan / dua sumbu", "mode": ["dua_sumbu", "berdampingan"]},
}
POSITIVE_TYPES = {"stok", "arus", "indeks", "jumlah", "belum_pasti"}


def pair(a: dict, b: dict, sa: pd.Series | None = None, sb: pd.Series | None = None) -> dict:
    ua, ub = unit_text(a), unit_text(b)
    a_conf, b_conf = a["unit_status"] in CONFIRMED, b["unit_status"] in CONFIRMED
    reasons = []
    if a["indicator_id"] == b["indicator_id"]:
        return {"level": "absolut", "alasan": "Indikator yang sama.", **_meta("absolut")}
    same_unit = a_conf and b_conf and ua == ub
    conv = None
    if a_conf and b_conf and ua != ub:
        conv = find_conversion(ua, ub) or find_conversion(ub, ua)
    same_type = a["measure_type"] == b["measure_type"] and a["measure_type"] != "belum_pasti"
    same_concept = a["concept_id"] == b["concept_id"]

    if (same_unit or conv) and same_type and same_concept:
        txt = f"Konsep sama ({a['concept_name']}), jenis sama ({a['measure_type']}), satuan sama ({ua})"
        if conv:
            txt = f"Konsep & jenis sama; satuan dapat dikonversi via aturan eksplisit {conv.get('dari')}→{conv.get('ke')}"
        if not (a["measure_certain"] and b["measure_certain"]):
            txt += "; klasifikasi stok/arus masih indikatif"
        return {"level": "absolut", "alasan": txt + ".", **_meta("absolut")}

    if not a_conf:
        reasons.append(f"satuan {a['label_tampil']} belum terkonfirmasi")
    if not b_conf:
        reasons.append(f"satuan {b['label_tampil']} belum terkonfirmasi")
    if a_conf and b_conf and ua != ub:
        reasons.append(f"satuan berbeda ({ua} vs {ub}) tanpa aturan konversi eksplisit")
    if not same_concept:
        reasons.append(f"konsep berbeda ({a['concept_name']} vs {b['concept_name']})")
    if a["measure_type"] != b["measure_type"]:
        reasons.append(f"jenis ukuran berbeda ({a['measure_type']} vs {b['measure_type']})")

    if a["measure_type"] == "rasio" and b["measure_type"] == "rasio" and ua == "%" and ub == "%":
        return {"level": "perubahan", "alasan": "Keduanya rasio %, tetapi " + "; ".join(reasons) +
                ". Bandingkan perubahan dalam pp atau tampilkan berdampingan.", **_meta("perubahan")}

    def positive(s, info):
        if info["measure_type"] not in POSITIVE_TYPES:
            return False
        return s is None or (s.dropna() > 0).all()

    if positive(sa, a) and positive(sb, b):
        return {"level": "indeks", "alasan": _cap("; ".join(reasons)) +
                ". Bandingkan pola pertumbuhan dengan indeks basis 100 atau % perubahan, bukan besaran.",
                **_meta("indeks")}
    extra = ""
    if sa is not None and not positive(sa, a) and a["measure_type"] in POSITIVE_TYPES:
        extra = f" {a['label_tampil']} memuat nilai ≤ 0 sehingga indeks basis 100 tidak valid."
    if sb is not None and not positive(sb, b) and b["measure_type"] in POSITIVE_TYPES:
        extra += f" {b['label_tampil']} memuat nilai ≤ 0 sehingga indeks basis 100 tidak valid."
    return {"level": "berdampingan", "alasan": _cap("; ".join(reasons)) + "." + extra +
            " Gunakan dua sumbu berlabel atau grafik berdampingan.", **_meta("berdampingan")}


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _meta(level: str) -> dict:
    return {"label": LEVELS[level]["label"], "skor": LEVELS[level]["skor"], "mode": LEVELS[level]["mode"]}


def group_level(infos: list[dict], series: dict[str, pd.Series] | None = None) -> dict:
    """Level kompatibilitas terendah di antara semua pasangan (untuk >2 indikator)."""
    series = series or {}
    worst = None
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            p = pair(infos[i], infos[j], series.get(infos[i]["indicator_id"]), series.get(infos[j]["indicator_id"]))
            if worst is None or p["skor"] < worst["skor"]:
                worst = p
    return worst or {"level": "absolut", "alasan": "Satu indikator.", **_meta("absolut")}


def matrix(catalog: pd.DataFrame, ids: list[str], series: dict[str, pd.Series] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kembalikan (matriks skor, matriks alasan) dengan label 'Indikator — Subsektor'."""
    series = series or {}
    infos = [catalog.loc[catalog["indicator_id"] == i].iloc[0].to_dict() for i in ids]
    labels = [f"{x['label_tampil']} — {x['sheet']}" for x in infos]
    score = pd.DataFrame(index=labels, columns=labels, dtype=float)
    why = pd.DataFrame(index=labels, columns=labels, dtype=object)
    for i, a in enumerate(infos):
        for j, b in enumerate(infos):
            p = pair(a, b, series.get(a["indicator_id"]), series.get(b["indicator_id"]))
            score.iloc[i, j] = p["skor"]
            why.iloc[i, j] = f"{p['label']}: {p['alasan']}"
    return score, why
