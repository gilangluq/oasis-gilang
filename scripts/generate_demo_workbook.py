"""Bangkitkan workbook DEMO SIMULASI dengan struktur yang sama seperti workbook sumber.

Seluruh nilai adalah ACAK/SINTETIS (bukan data OJK) sehingga aman untuk repositori
dan demo publik. Struktur sheet, nama field, dan Kamus_Data mengikuti
``data/demo/kamus_template.json``.

Pemakaian:
    python scripts/generate_demo_workbook.py [--out data/demo/oasis_demo_simulasi.xlsx] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "data" / "demo" / "kamus_template.json"
RELATIONS = json.loads((ROOT / "config" / "relations.json").read_text(encoding="utf-8"))["aturan"]

# Level awal sintetis (angka bulat rekaan) per kata kunci label
ANCHORS = [
    ("rasio npl", 3.0), ("rasio npf", 3.5), ("rasio car", 24.0), ("rasio ldr", 85.0), ("rasio nim", 4.6),
    ("rasio bopo", 85.0), ("rasio roa", 2.0), ("rasio roe", 12.0), ("rasio far", 82.0), ("rasio rasio gr", 2.1),
    ("combined", 90.0), ("rbc", 400.0), ("beban usaha", 22.0), ("kecukupan investasi", 140.0), ("rasio klaim", 60.0),
    ("likuiditas", 140.0), ("rasio investasi", 97.0), ("roi r+u", 0.0), ("roi r", 0.0), ("roa", 0.0),
    ("ihsg", 6000.0), ("perusahaan tercatat", 720.0), ("kapitalisasi", 7000.0), ("volume", 400.0),
    ("nilai perdagangan", 300.0), ("frekuensi", 25000.0),
    ("total aset", 500000.0), ("total kredit", 300000.0), ("dpk", 350000.0), ("giro", 100000.0),
    ("tabungan", 120000.0), ("deposito", 130000.0), ("laba", 20000.0), ("bukan investasi", 60000.0),
    ("total investasi", 400000.0), ("sbn", 50000.0), ("obligasi", 40000.0), ("aset neto", 110000.0),
    ("piutang", 360000.0), ("klaim", 3000.0), ("premi", 4000.0),
]


def anchor(label: str) -> float:
    l = label.lower()
    for k, v in ANCHORS:
        if k in l:
            return v
    return 1000.0


def simulate(label: str, unit: str, desc: str, n: int, rng: np.random.Generator) -> np.ndarray:
    a = anchor(label)
    l = label.lower()
    if "laba" in l or l.startswith("roi") or l == "roa":
        # arus kumulatif tahun berjalan (reset Januari)
        monthly = np.abs(rng.normal(a / 12 if a else 0.5, (a / 12 if a else 0.5) * 0.25, n))
        out = np.zeros(n)
        for i in range(n):
            out[i] = monthly[i] + (out[i - 1] if i % 12 else 0)
        return out
    if unit == "%" or "rbc" in l or "combined" in l:
        x = np.empty(n)
        x[0] = a
        for i in range(1, n):
            x[i] = x[i - 1] + 0.15 * (a - x[i - 1]) + rng.normal(0, max(a * 0.01, 0.03))
        return x
    if "perusahaan tercatat" in l:
        return np.round(a + np.cumsum(rng.integers(0, 6, n)))
    if "selama periode" in desc.lower():
        return np.abs(a * (1 + rng.normal(0, 0.18, n)))
    growth = rng.normal(0.006, 0.01, n)
    return a * np.exp(np.cumsum(growth))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "demo" / "oasis_demo_simulasi.xlsx"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", default="2021-01-31")
    ap.add_argument("--end", default="2026-06-30")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    tpl = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    kamus = pd.DataFrame(tpl["kamus"])
    periods = pd.date_range(args.start, args.end, freq="ME")
    n = len(periods)
    sheets, dict_rows, meta_rows = {}, [], []
    for sheet, grp in kamus.groupby("sheet", sort=False):
        df = pd.DataFrame({"Periode": periods.strftime("%Y%m%d").astype(int)})
        for r in grp.itertuples():
            if r.field_asli_kamus == "Periode":
                continue
            df[r.field_asli_kamus] = simulate(r.label_standar, r.unit_raw, r.deskripsi, n, rng)
        # ---- jaga konsistensi identitas sesuai config/relations.json ----
        lab2field = {" ".join(str(r.label_standar).split()).lower(): r.field_asli_kamus for r in grp.itertuples()}
        for rule in RELATIONS:
            if rule["sheet"] != sheet:
                continue
            f = lambda lab: lab2field.get(" ".join(lab.split()).lower())
            tgt = f(rule["target"])
            if rule["jenis"] == "penjumlahan":
                df[tgt] = df[[f(c) for c in rule["komponen"]]].sum(axis=1)
            elif rule["jenis"] == "rasio":
                df[tgt] = df[f(rule["pembilang"])] / df[f(rule["penyebut"])] * rule.get("skala", 100)
            elif rule["jenis"] == "komponen_parsial":
                df[tgt] = df[[f(c) for c in rule["komponen"]]].sum(axis=1) * 1.18
        # ---- sisipkan kasus uji kualitas & anomali (terdokumentasi) ----
        num_cols = [c for c in df.columns if c != "Periode"]
        if sheet == "BPR":
            npl = [c for c in num_cols if "npl" in c.lower()][0]
            df.loc[n - 4, npl] += 2.5  # lonjakan untuk demo anomali
            roe = [c for c in num_cols if "roe" in c.lower()][0]
            df.loc[n - 3:, roe] = np.nan  # nilai hilang untuk demo laporan kualitas
        if sheet == "Bank Umum":
            car = [c for c in num_cols if "car" in c.lower()][0]
            df.loc[n - 2:, car] -= 4  # penurunan tajam CAR
        sheets[sheet] = df
        for r in grp.itertuples():
            s = df[r.field_asli_kamus]
            dict_rows.append([len(dict_rows) + 1, sheet, r.domain, r.field_asli_kamus, r.label_standar, r.deskripsi,
                              r.tipe_data, r.unit_raw, r.frekuensi, periods[0].strftime("%Y-%m-%d"),
                              periods[-1].strftime("%Y-%m-%d"), int(s.isna().sum()),
                              str(s.dropna().iloc[0]) if r.field_asli_kamus == "Periode" else round(float(s.dropna().iloc[0]), 4)])
        dm = next((m for m in tpl["metadata"] if m["sheet"] == sheet), {})
        meta_rows.append([len(meta_rows) + 1, sheet, dm.get("domain", grp.iloc[0]["domain"]), "Bulanan",
                          dm.get("grain", "Agregat subsektor per akhir bulan"), periods[0].strftime("%Y-%m-%d"),
                          periods[-1].strftime("%Y-%m-%d"), n, len(grp), "Periode",
                          "DATA SIMULASI (acak) untuk demo OASIS — bukan data OJK."])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for sheet, df in sheets.items():
            df.to_excel(xw, sheet_name=sheet, index=False)
        meta = pd.DataFrame(meta_rows, columns=["No", "Nama Sheet", "Domain/Subsektor", "Frekuensi", "Grain/Level Observasi",
                                                "Periode Awal", "Periode Akhir", "Jumlah Observasi", "Jumlah Variabel",
                                                "Primary Time Key", "Catatan Kualitas"])
        pd.DataFrame([["METADATA DATASET KEUANGAN - DEMO SIMULASI"], ["Seluruh nilai acak/sintetis untuk demo; bukan data OJK."], [None]]).to_excel(
            xw, sheet_name="Metadata_Sheet", index=False, header=False)
        meta.to_excel(xw, sheet_name="Metadata_Sheet", index=False, startrow=3)
        kd = pd.DataFrame(dict_rows, columns=["No", "Sheet", "Domain/Subsektor", "Nama Field Asli", "Label Standar", "Deskripsi",
                                              "Tipe Data", "Unit", "Frekuensi", "Mulai Tersedia", "Akhir Tersedia",
                                              "Jumlah Null", "Contoh Nilai"])
        pd.DataFrame([["KAMUS DATA - DEMO SIMULASI"], ["Struktur mengikuti kamus sumber; nilai contoh sintetis."], [None]]).to_excel(
            xw, sheet_name="Kamus_Data", index=False, header=False)
        kd.to_excel(xw, sheet_name="Kamus_Data", index=False, startrow=3)
    print(f"Workbook demo simulasi ditulis ke {out}")


if __name__ == "__main__":
    main()
