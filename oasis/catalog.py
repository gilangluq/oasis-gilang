"""Katalog metadata OASIS.

Membaca sheet ``Kamus_Data`` dan ``Metadata_Sheet`` dari workbook sumber dan
menjadikannya sumber kebenaran katalog indikator. Modul ini TIDAK membuat field
baru: setiap indikator di katalog berasal dari satu baris Kamus_Data.

Keluaran utama: :func:`build_catalog` -> ``pandas.DataFrame`` satu baris per
indikator (selain field kunci ``Periode``).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

DICT_SHEET_CANDIDATES = ["Kamus_Data", "Kamus Data", "Data Dictionary"]
META_SHEET_CANDIDATES = ["Metadata_Sheet", "Metadata Sheet", "Metadata"]

DICT_COLUMNS = {
    "No": "no",
    "Sheet": "sheet",
    "Domain/Subsektor": "domain",
    "Nama Field Asli": "field_asli_kamus",
    "Label Standar": "label_standar",
    "Deskripsi": "deskripsi",
    "Tipe Data": "tipe_data",
    "Unit": "unit_raw",
    "Frekuensi": "frekuensi",
    "Mulai Tersedia": "mulai_kamus",
    "Akhir Tersedia": "akhir_kamus",
    "Jumlah Null": "null_kamus",
    "Contoh Nilai": "contoh",
}
META_COLUMNS = {
    "No": "no",
    "Nama Sheet": "sheet",
    "Domain/Subsektor": "domain",
    "Frekuensi": "frekuensi",
    "Grain/Level Observasi": "grain",
    "Periode Awal": "periode_awal",
    "Periode Akhir": "periode_akhir",
    "Jumlah Observasi": "jumlah_observasi",
    "Jumlah Variabel": "jumlah_variabel",
    "Primary Time Key": "time_key",
    "Catatan Kualitas": "catatan_kualitas",
}

UNIT_STATUS_LABEL = {
    "terkonfirmasi_kamus": "Terkonfirmasi (kamus)",
    "label_sumber": "Sesuai label sumber",
    "dikonfirmasi_pengguna": "Dikonfirmasi pengguna",
    "belum_terkonfirmasi": "Satuan belum terkonfirmasi",
    "kunci": "Kunci waktu",
}

MEASURE_LABEL = {
    "stok": "Stok (posisi akhir periode)",
    "arus": "Arus (selama periode)",
    "rasio": "Rasio (%)",
    "indeks": "Indeks",
    "jumlah": "Jumlah/hitungan (posisi)",
    "belum_pasti": "Belum pasti",
}


# ---------------------------------------------------------------------------
# Utilitas teks
# ---------------------------------------------------------------------------
def collapse_ws(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s if s is not None else "")).strip()


def norm_key(s: Any) -> str:
    """Kunci pembanding: huruf kecil + spasi dirapatkan."""
    return collapse_ws(s).casefold()


def sheet_suffix(sheet: str) -> str:
    """Akhiran nama subsektor yang sering ditempel pada label (mis. 'Asuransi Jiwa')."""
    s = collapse_ws(sheet)
    s = re.sub(r"^perusahaan\s+", "", s, flags=re.I)
    return s.casefold()


def normalize_label(label: str, sheet: str) -> str:
    """Normalisasi label untuk pencocokan konsep: tanpa kurung, tanpa akhiran subsektor."""
    s = norm_key(label)
    s = re.sub(r"\([^)]*\)", " ", s)
    s = collapse_ws(s)
    suf = sheet_suffix(sheet)
    if suf and s.endswith(" " + suf):
        s = s[: -len(suf) - 1]
    return collapse_ws(s)


def slugify(s: str) -> str:
    s = norm_key(s)
    s = re.sub(r"[^a-z0-9+]+", "_", s)
    return s.strip("_")


# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
def load_json(name: str) -> dict:
    path = CONFIG_DIR / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class ConceptConfig:
    subsektor: dict[str, list[str]]
    kelompok: dict[str, list[str]]
    konsep: list[dict]
    ambigu: dict[str, list[str]]
    by_id: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "ConceptConfig":
        d = load_json("concepts.json")
        cfg = cls(
            subsektor=d.get("subsektor", {}),
            kelompok=d.get("kelompok_subsektor", {}),
            konsep=d.get("konsep", []),
            ambigu=d.get("sinonim_ambigu", {}),
        )
        cfg.by_id = {k["id"]: k for k in cfg.konsep}
        return cfg

    def concept_for_label(self, label_norm: str) -> dict | None:
        for k in self.konsep:
            if label_norm in k.get("labels", []):
                return k
        return None


# ---------------------------------------------------------------------------
# Membaca sheet kamus/metadata
# ---------------------------------------------------------------------------
def _find_sheet(sheet_names: list[str], candidates: list[str]) -> str | None:
    keys = {norm_key(s): s for s in sheet_names}
    for c in candidates:
        if norm_key(c) in keys:
            return keys[norm_key(c)]
    return None


def _read_table_with_header(xls: pd.ExcelFile, sheet: str, must_have: str, colmap: dict) -> pd.DataFrame:
    raw = xls.parse(sheet, header=None)
    header_row = None
    for i in range(min(len(raw), 15)):
        vals = [collapse_ws(v) for v in raw.iloc[i].tolist()]
        if must_have in vals:
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"Sheet '{sheet}' tidak memiliki baris header yang memuat '{must_have}'.")
    header = [collapse_ws(v) for v in raw.iloc[header_row].tolist()]
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = header
    df = df.loc[:, [c for c in df.columns if c and c != "nan"]]
    df = df.dropna(how="all")
    keep = {c: colmap[c] for c in df.columns if c in colmap}
    df = df[list(keep)].rename(columns=keep).reset_index(drop=True)
    return df


def read_dictionary(xls: pd.ExcelFile) -> tuple[pd.DataFrame, pd.DataFrame | None, list[dict]]:
    """Kembalikan (kamus, metadata_sheet, catatan_masalah)."""
    issues: list[dict] = []
    dsheet = _find_sheet(xls.sheet_names, DICT_SHEET_CANDIDATES)
    if dsheet is None:
        raise ValueError(
            "Sheet kamus data (Kamus_Data) tidak ditemukan. OASIS membutuhkan kamus sebagai sumber kebenaran katalog."
        )
    kamus = _read_table_with_header(xls, dsheet, "Nama Field Asli", DICT_COLUMNS)
    missing_cols = [v for v in DICT_COLUMNS.values() if v not in kamus.columns]
    for c in missing_cols:
        issues.append({"tingkat": "Peringatan", "sheet": dsheet, "field": c, "pemeriksaan": "Kolom kamus",
                       "temuan": f"Kolom kamus '{c}' tidak ditemukan.", "rekomendasi": "Lengkapi kamus data."})
        kamus[c] = None
    kamus = kamus[kamus["field_asli_kamus"].notna()].copy()
    for c in ["sheet", "domain", "field_asli_kamus", "label_standar", "deskripsi", "tipe_data", "unit_raw", "frekuensi"]:
        kamus[c] = kamus[c].map(lambda v: "" if pd.isna(v) else str(v))
    kamus["sheet"] = kamus["sheet"].map(collapse_ws)

    meta = None
    msheet = _find_sheet(xls.sheet_names, META_SHEET_CANDIDATES)
    if msheet is not None:
        try:
            meta = _read_table_with_header(xls, msheet, "Nama Sheet", META_COLUMNS)
            meta = meta[meta["sheet"].notna()].copy()
            meta["sheet"] = meta["sheet"].map(collapse_ws)
        except ValueError as e:  # pragma: no cover - struktur tak terduga
            issues.append({"tingkat": "Peringatan", "sheet": msheet, "field": "", "pemeriksaan": "Metadata_Sheet",
                           "temuan": str(e), "rekomendasi": "Periksa struktur Metadata_Sheet."})
    else:
        issues.append({"tingkat": "Info", "sheet": "Metadata_Sheet", "field": "", "pemeriksaan": "Metadata_Sheet",
                       "temuan": "Sheet Metadata_Sheet tidak ditemukan; ringkasan per sheet dihitung dari data.",
                       "rekomendasi": "Opsional: sertakan Metadata_Sheet."})
    return kamus, meta, issues


# ---------------------------------------------------------------------------
# Klasifikasi satuan & jenis ukuran
# ---------------------------------------------------------------------------
def classify_unit(unit_raw: str) -> tuple[str | None, str, str]:
    """Kembalikan (unit_tampil, status, catatan)."""
    u = collapse_ws(unit_raw)
    ul = u.casefold()
    if not u or "tidak tercantum" in ul or "belum" in ul:
        return None, "belum_terkonfirmasi", "Kamus: satuan tidak tercantum pada header sumber."
    if "tanggal" in ul or "yyyymmdd" in ul:
        return "Tanggal", "kunci", "Kunci waktu."
    if "sesuai label sumber" in ul:
        base = collapse_ws(re.sub(r"\(.*?\)", "", u))
        note = f"Kamus: '{u}'."
        if base.lower() == "rp m":
            note += " Arti 'M' (juta/miliar) tidak didefinisikan di kamus; tidak dikonversi ke satuan lain."
        return base, "label_sumber", note
    return u, "terkonfirmasi_kamus", f"Kamus: '{u}'."


def classify_measure(desc: str, unit: str | None, unit_status: str, label: str) -> tuple[str, bool, str]:
    """Kembalikan (jenis, pasti, alasan) berdasarkan definisi kamus.

    Aturan bersifat eksplisit dan dapat diperiksa; bila tidak pasti, ditandai.
    """
    d = (desc or "").casefold()
    lab = (label or "").casefold()
    if unit == "%":
        return "rasio", True, "Satuan kamus '%'."
    if unit and "indeks" in unit.casefold():
        return "indeks", True, "Satuan kamus 'Poin indeks'."
    if "selama periode" in d:
        return "arus", True, "Deskripsi kamus: 'selama periode'."
    if unit and unit.casefold() == "perusahaan":
        return "jumlah", True, "Satuan kamus 'Perusahaan' (hitungan posisi)."
    if "akhir periode" in d:
        return "stok", True, "Deskripsi kamus: 'pada akhir periode'."
    if "periode berjalan" in d:
        return "arus", False, ("Deskripsi kamus: 'periode berjalan'. Basis bulanan vs kumulatif tahun berjalan "
                               "tidak dinyatakan; jangan dijumlahkan antarbulan sebelum dikonfirmasi.")
    if "rasio" in lab or "rbc" in lab or "ratio" in lab:
        return "belum_pasti", False, ("Nama tampak seperti rasio, tetapi satuan tidak tercantum di kamus. "
                                      "Perubahan dinyatakan sebagai selisih dalam satuan sumber, bukan pp.")
    if any(k in d for k in ["total nilai", "nilai investasi", "nilai penempatan", "total aset", "pos bukan investasi"]):
        return "stok", False, "Indikatif stok (nilai posisi/portofolio); kamus tidak menyebut 'akhir periode'."
    return "belum_pasti", False, "Kamus tidak menyatakan apakah stok atau arus."


# ---------------------------------------------------------------------------
# Katalog
# ---------------------------------------------------------------------------
def _fmt_date(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    try:
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    except Exception:
        return collapse_ws(v)


def build_catalog(kamus: pd.DataFrame, concepts: ConceptConfig | None = None) -> pd.DataFrame:
    concepts = concepts or ConceptConfig.load()
    rows = []
    for _, r in kamus.iterrows():
        unit, status, unote = classify_unit(r["unit_raw"])
        if status == "kunci":
            continue  # Periode = kunci waktu, bukan indikator
        label = collapse_ws(r["label_standar"]) or collapse_ws(r["field_asli_kamus"])
        mtype, mcertain, mnote = classify_measure(r["deskripsi"], unit, status, label)
        lnorm = normalize_label(label, r["sheet"])
        concept = concepts.concept_for_label(lnorm)
        rows.append(
            {
                "indicator_id": f"{slugify(r['sheet'])}::{slugify(label)}",
                "sheet": r["sheet"],
                "domain": r["domain"],
                "field_asli_kamus": r["field_asli_kamus"],
                "label_standar": r["label_standar"],
                "label_tampil": label,
                "label_norm": lnorm,
                "deskripsi": r["deskripsi"],
                "tipe_data": r["tipe_data"],
                "unit_raw": r["unit_raw"],
                "unit": unit,
                "unit_status": status,
                "unit_note": unote,
                "measure_type": mtype,
                "measure_certain": mcertain,
                "measure_note": mnote,
                "frekuensi": r["frekuensi"],
                "mulai_kamus": _fmt_date(r.get("mulai_kamus")),
                "akhir_kamus": _fmt_date(r.get("akhir_kamus")),
                "null_kamus": r.get("null_kamus"),
                "contoh": r.get("contoh"),
                "concept_id": concept["id"] if concept else f"label:{lnorm}",
                "concept_name": concept["nama"] if concept else label,
                "arah_perhatian": concept.get("arah_perhatian") if concept else None,
            }
        )
    cat = pd.DataFrame(rows)
    if not cat.empty and cat["indicator_id"].duplicated().any():
        # jaga keunikan id tanpa mengubah label
        cat["indicator_id"] = cat["indicator_id"] + cat.groupby("indicator_id").cumcount().map(
            lambda i: "" if i == 0 else f"_{i+1}")
    return cat


def unit_display(row: pd.Series | dict) -> str:
    """Teks satuan untuk ditampilkan di nilai/sumbu."""
    u = row.get("unit") if isinstance(row, dict) else row["unit"]
    st = row.get("unit_status") if isinstance(row, dict) else row["unit_status"]
    if st == "belum_terkonfirmasi" or not u:
        return "satuan belum terkonfirmasi"
    return u


def apply_unit_confirmations(catalog: pd.DataFrame, confirmations: dict[str, dict]) -> pd.DataFrame:
    """Terapkan konfirmasi satuan yang dinyatakan pengguna (dicatat dan dapat diaudit)."""
    if not confirmations:
        return catalog
    cat = catalog.copy()
    for iid, conf in confirmations.items():
        m = cat["indicator_id"] == iid
        if not m.any() or not conf.get("unit"):
            continue
        cat.loc[m, "unit"] = conf["unit"]
        cat.loc[m, "unit_status"] = "dikonfirmasi_pengguna"
        cat.loc[m, "unit_note"] = (
            f"Dikonfirmasi pengguna '{conf.get('oleh', '-')}' pada {conf.get('waktu', '-')}: {conf['unit']}. "
            f"Dasar: {conf.get('dasar', '-')}"
        )
        if conf["unit"] == "%":
            cat.loc[m, "measure_type"] = "rasio"
            cat.loc[m, "measure_certain"] = True
            cat.loc[m, "measure_note"] = "Satuan '%' dikonfirmasi pengguna."
    return cat


def load_unit_conversions() -> list[dict]:
    try:
        return load_json("unit_conversions.json").get("konversi", [])
    except FileNotFoundError:  # pragma: no cover
        return []


def find_conversion(from_unit: str | None, to_unit: str | None) -> dict | None:
    if not from_unit or not to_unit:
        return None
    for c in load_unit_conversions():
        if norm_key(c.get("dari")) == norm_key(from_unit) and norm_key(c.get("ke")) == norm_key(to_unit):
            return c
    return None
