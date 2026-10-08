"""Validasi & penyiapan data OASIS.

Tanggung jawab modul ini:
1. Membaca workbook (sheet data per subsektor + kamus).
2. Merekonsiliasi field kamus dengan kolom sheet (cocok persis / normalisasi /
   posisi) dan mencatat metode pemetaannya agar dapat diaudit.
3. Mem-parsing ``Periode`` (YYYYMMDD -> tanggal akhir bulan).
4. Menjalankan pemeriksaan kualitas data dan menyusun laporan.
5. Menghasilkan data panjang (tidy) + asal data (provenance).
"""
from __future__ import annotations

import difflib
import hashlib
import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import __version__
from .catalog import (
    ConceptConfig,
    apply_unit_confirmations,
    build_catalog,
    collapse_ws,
    load_json,
    norm_key,
    normalize_label,
    read_dictionary,
)

TZ = ZoneInfo("Asia/Jakarta")
STATUS_DATA = ["Simulasi", "Asli (internal - jangan dipublikasikan)", "Publik", "Belum dinyatakan"]


# ---------------------------------------------------------------------------
# Periode
# ---------------------------------------------------------------------------
def parse_periode_value(v: Any) -> tuple[pd.Timestamp | None, str | None]:
    """Parse satu nilai Periode. Kembalikan (tanggal_akhir_bulan, catatan)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None, "kosong"
    ts = None
    if isinstance(v, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(v)
    else:
        s = str(v).strip()
        if re.fullmatch(r"\d+(\.0+)?", s):
            s = s.split(".")[0]
            try:
                if len(s) == 8:
                    ts = pd.Timestamp(datetime.strptime(s, "%Y%m%d"))
                elif len(s) == 6:
                    ts = pd.Timestamp(datetime.strptime(s, "%Y%m"))
            except ValueError:
                return None, f"angka '{s}' bukan tanggal YYYYMMDD yang valid"
        else:
            try:
                ts = pd.Timestamp(s)
            except (ValueError, TypeError):
                return None, f"format '{s}' tidak dikenali"
    if ts is None:
        return None, f"format '{v}' tidak dikenali"
    me = (ts + pd.offsets.MonthEnd(0)).normalize()
    note = None if me == ts.normalize() else f"{ts.date()} disesuaikan ke akhir bulan {me.date()}"
    return me, note


def parse_periode_series(s: pd.Series) -> tuple[pd.Series, list[str], list[str]]:
    out, invalid, adjusted = [], [], []
    for v in s.tolist():
        ts, note = parse_periode_value(v)
        out.append(ts)
        if ts is None:
            invalid.append(f"{v!r}: {note}")
        elif note:
            adjusted.append(note)
    return pd.Series(pd.to_datetime(out), index=s.index), invalid, adjusted


# ---------------------------------------------------------------------------
# Struktur hasil
# ---------------------------------------------------------------------------
@dataclass
class Issue:
    tingkat: str  # Kritis | Peringatan | Info
    sheet: str
    field: str
    pemeriksaan: str
    temuan: str
    rekomendasi: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class OasisDataset:
    catalog: pd.DataFrame
    long: pd.DataFrame  # kolom: indicator_id, periode, nilai
    wide: dict[str, pd.DataFrame]  # per sheet: index periode, kolom indicator_id
    mapping: pd.DataFrame
    sheet_summary: pd.DataFrame
    issues: list[Issue]
    rule_results: pd.DataFrame
    flags: dict[str, list[dict]]  # indicator_id -> daftar flag kualitas
    metadata_sheet: pd.DataFrame | None
    provenance: dict
    base_catalog: pd.DataFrame = field(default=None)  # sebelum konfirmasi satuan

    # ---- akses -----------------------------------------------------------
    def info(self, iid: str) -> dict:
        r = self.catalog.loc[self.catalog["indicator_id"] == iid]
        if r.empty:
            raise KeyError(iid)
        return r.iloc[0].to_dict()

    def series(self, iid: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
               dropna: bool = True) -> pd.Series:
        sheet = self.info(iid)["sheet"]
        s = self.wide[sheet][iid].copy() if iid in self.wide.get(sheet, {}) else pd.Series(dtype=float)
        if start is not None:
            s = s[s.index >= start]
        if end is not None:
            s = s[s.index <= end]
        s.name = iid
        return s.dropna() if dropna else s

    def period_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.long["periode"].min(), self.long["periode"].max()

    def with_unit_confirmations(self, confirmations: dict[str, dict]) -> "OasisDataset":
        base = self.base_catalog if self.base_catalog is not None else self.catalog
        new = OasisDataset(**{**self.__dict__})
        new.base_catalog = base
        new.catalog = apply_unit_confirmations(base, confirmations)
        return new

    def issues_df(self) -> pd.DataFrame:
        return pd.DataFrame([i.as_dict() for i in self.issues],
                            columns=["tingkat", "sheet", "field", "pemeriksaan", "temuan", "rekomendasi"])


# ---------------------------------------------------------------------------
# Rekonsiliasi kamus <-> kolom sheet
# ---------------------------------------------------------------------------
def _col_norm(col: str, sheet: str) -> str:
    return normalize_label(col, sheet)


def reconcile_fields(sheet: str, kamus_fields: list[dict], sheet_cols: list[str]) -> list[dict]:
    """Petakan setiap field kamus ke kolom sheet.

    Urutan metode: (1) cocok persis, (2) cocok setelah normalisasi (tanpa satuan
    dalam kurung & akhiran subsektor), (3) posisi urutan kolom bila jumlah field
    tersisa sama. Metode (3) selalu diberi status 'perlu konfirmasi'.
    """
    results: dict[int, dict] = {}
    remaining_cols = list(sheet_cols)

    # (1) persis
    for i, f in enumerate(kamus_fields):
        for c in remaining_cols:
            if norm_key(c) == norm_key(f["field_asli_kamus"]):
                results[i] = {"kolom_sheet": c, "metode": "cocok persis", "keyakinan": 1.0}
                remaining_cols.remove(c)
                break
    # (2) normalisasi
    for i, f in enumerate(kamus_fields):
        if i in results:
            continue
        fn = normalize_label(f["field_asli_kamus"], sheet)
        for c in remaining_cols:
            if _col_norm(c, sheet) == fn:
                results[i] = {"kolom_sheet": c, "metode": "cocok setelah normalisasi", "keyakinan": 0.9}
                remaining_cols.remove(c)
                break
    # (3) posisi untuk sisa (hanya bila jumlah sama)
    unmatched = [i for i in range(len(kamus_fields)) if i not in results]
    if unmatched and len(unmatched) == len(remaining_cols):
        ordered_cols = [c for c in sheet_cols if c in remaining_cols]
        for i, c in zip(unmatched, ordered_cols):
            sim = difflib.SequenceMatcher(None, normalize_label(kamus_fields[i]["field_asli_kamus"], sheet),
                                          _col_norm(c, sheet)).ratio()
            results[i] = {"kolom_sheet": c, "metode": "posisi kolom (perlu konfirmasi)",
                          "keyakinan": round(0.5 * sim + 0.3, 2)}
        remaining_cols = []

    out = []
    for i, f in enumerate(kamus_fields):
        r = results.get(i, {"kolom_sheet": None, "metode": "tidak ditemukan", "keyakinan": 0.0})
        out.append({**f, **r})
    for c in remaining_cols:
        out.append({"indicator_id": None, "field_asli_kamus": None, "label_tampil": None, "kolom_sheet": c,
                    "metode": "kolom tidak ada di kamus (diabaikan)", "keyakinan": 0.0})
    return out


# ---------------------------------------------------------------------------
# Heuristik pola data
# ---------------------------------------------------------------------------
def detect_ytd_pattern(s: pd.Series) -> dict | None:
    """Deteksi pola kumulatif tahun berjalan (reset Januari). Heuristik, belum konfirmasi."""
    s = s.dropna()
    if len(s) < 24:
        return None
    years_ok, years_tested = 0, 0
    for y in sorted(set(s.index.year)):
        ys = s[s.index.year == y]
        prev_dec = s[(s.index.year == y - 1) & (s.index.month == 12)]
        if len(ys) < 6 or prev_dec.empty or ys.index[0].month != 1:
            continue
        years_tested += 1
        dec = float(prev_dec.iloc[0])
        jan = float(ys.iloc[0])
        diffs = np.diff(ys.values)
        rising = (diffs > 0).mean() if len(diffs) else 0
        if dec > 0 and jan < 0.5 * dec and rising >= 0.8:
            years_ok += 1
    if years_tested >= 2 and years_ok >= max(2, years_tested - 1):
        return {"tahun_uji": years_tested, "tahun_cocok": years_ok}
    return None


def ytd_to_monthly(s: pd.Series) -> pd.Series:
    """De-kumulasi heuristik: Januari = nilai, bulan lain = selisih terhadap bulan sebelumnya pada tahun yang sama."""
    s = s.dropna().sort_index()
    out = s.copy()
    for i in range(1, len(s)):
        cur, prev = s.index[i], s.index[i - 1]
        if cur.year == prev.year and (cur.to_period("M") - prev.to_period("M")).n == 1:
            out.iloc[i] = s.iloc[i] - s.iloc[i - 1]
        elif cur.month != 1:
            out.iloc[i] = np.nan  # ada bulan hilang -> tak bisa didekumulasi
    return out


# ---------------------------------------------------------------------------
# Uji aturan eksplisit
# ---------------------------------------------------------------------------
def run_relation_rules(catalog: pd.DataFrame, wide: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rules = load_json("relations.json").get("aturan", [])

    def find_iid(sheet: str, label: str) -> str | None:
        m = catalog[(catalog["sheet"] == sheet) & (catalog["label_tampil"].map(norm_key) == norm_key(label))]
        return None if m.empty else m.iloc[0]["indicator_id"]

    rows = []
    for r in rules:
        sheet = r["sheet"]
        target = find_iid(sheet, r["target"])
        refs = r.get("komponen") or [x for x in [r.get("pembilang"), r.get("penyebut")] if x]
        ref_ids = [find_iid(sheet, x) for x in refs]
        base = {"id": r["id"], "sheet": sheet, "jenis": r["jenis"], "target": r["target"],
                "referensi": ", ".join(refs), "target_id": target, "referensi_id": ref_ids, "dasar": r["dasar"]}
        if target is None or any(x is None for x in ref_ids) or sheet not in wide:
            rows.append({**base, "status": "tidak dapat diuji", "n": 0, "lolos_pct": np.nan,
                         "detail": "Field aturan tidak ditemukan di katalog/sheet."})
            continue
        df = wide[sheet]
        if r["jenis"] == "definisi_tak_teruji":
            rows.append({**base, "status": "tidak dapat diuji", "n": 0, "lolos_pct": np.nan,
                         "detail": "Hubungan definisional; field pembanding tidak tersedia di data."})
            continue
        t = df[target]
        if r["jenis"] == "penjumlahan":
            calc = df[ref_ids].sum(axis=1, min_count=len(ref_ids))
            ok = (calc - t).abs() <= r.get("toleransi_relatif", 0.01) * t.abs()
            detail_val = ((calc - t).abs() / t.abs()).median() * 100
            detail = f"Median selisih relatif {detail_val:.3f}% (toleransi {r.get('toleransi_relatif', 0.01)*100:.1f}%)."
        elif r["jenis"] == "rasio":
            calc = df[ref_ids[0]] / df[ref_ids[1]] * r.get("skala", 100)
            ok = (calc - t).abs() <= r.get("toleransi_absolut", 0.5)
            med_calc = calc.median()
            detail = (f"Median hasil hitung {med_calc:,.4g} vs median nilai sumber {t.median():,.4g} "
                      f"(toleransi {r.get('toleransi_absolut', 0.5)} poin).")
        elif r["jenis"] == "komponen_parsial":
            calc = df[ref_ids].sum(axis=1, min_count=len(ref_ids))
            ok = calc <= t * 1.0001
            detail = f"Jumlah komponen / total: median {(calc / t).median()*100:.1f}%."
        else:
            continue
        valid = ok[t.notna() & calc.notna()]
        n = int(valid.shape[0])
        pct = float(valid.mean() * 100) if n else np.nan
        status = "tidak dapat diuji" if n == 0 else ("lolos" if pct >= 95 else ("sebagian" if pct >= 50 else "gagal"))
        rows.append({**base, "status": status, "n": n, "lolos_pct": pct, "detail": detail})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Loader utama
# ---------------------------------------------------------------------------
def _read_data_sheet(xls: pd.ExcelFile, sheet: str) -> tuple[pd.DataFrame, dict]:
    raw = xls.parse(sheet, header=None)
    info = {"baris_mentah": int(raw.shape[0]), "kolom_mentah": int(raw.shape[1])}
    hdr = None
    for i in range(min(len(raw), 10)):
        if any(norm_key(v) == "periode" for v in raw.iloc[i].tolist()):
            hdr = i
            break
    if hdr is None:
        return pd.DataFrame(), {**info, "error": "Field 'Periode' tidak ditemukan pada 10 baris pertama."}
    header = [collapse_ws(v) if not (isinstance(v, float) and np.isnan(v)) else "" for v in raw.iloc[hdr].tolist()]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = header
    before_r, before_c = df.shape
    df = df.dropna(how="all")
    df = df.loc[:, [c for c in df.columns if c != ""]]
    df = df.dropna(axis=1, how="all")
    info.update({"baris_kosong_dibuang": int(before_r - df.shape[0]),
                 "kolom_kosong_dibuang": int(before_c - df.shape[1]), "baris_header": hdr + 1})
    if df.columns.duplicated().any():
        info["kolom_duplikat"] = df.columns[df.columns.duplicated()].tolist()
        df = df.loc[:, ~df.columns.duplicated()]
    return df.reset_index(drop=True), info


CONSOLIDATE_CHECKS = {"Nama field kamus vs header sheet", "Konflik satuan kamus vs header"}


def _consolidate_issues(issues: list[Issue]) -> list[Issue]:
    """Gabungkan temuan berulang per sheet agar laporan mudah dibaca (rincian tetap ada di tabel pemetaan)."""
    keep, groups = [], {}
    for i in issues:
        if i.pemeriksaan in CONSOLIDATE_CHECKS:
            groups.setdefault((i.tingkat, i.sheet, i.pemeriksaan), []).append(i)
        else:
            keep.append(i)
    for (lvl, sheet, chk), items in groups.items():
        if len(items) == 1:
            keep.append(items[0])
            continue
        fields = [x.field for x in items]
        if chk == "Konflik satuan kamus vs header":
            units = sorted({u for x in items for u in re.findall(r"'\(([^)]*)\)'", x.temuan)})
            temuan = (f"{len(items)} field: header sheet mencantumkan satuan ({', '.join(units)}), sedangkan kamus "
                      f"menyatakan satuan tidak tercantum. OASIS mengikuti kamus → 'satuan belum terkonfirmasi'. "
                      f"Field: {', '.join(fields)}.")
        else:
            metode = sorted({m for x in items for m in re.findall(r"metode: (.+?) \(keyakinan", x.temuan)})
            temuan = (f"{len(items)} field memiliki nama kamus berbeda dari header sheet (metode: "
                      f"{'; '.join(metode)}). Field: {', '.join(fields)}. Rincian di tabel pemetaan.")
        keep.append(Issue(lvl, sheet, f"{len(items)} field", chk, temuan, items[0].rekomendasi))
    return keep


def load_workbook(file: bytes | str | io.BytesIO, filename: str = "workbook.xlsx",
                  status_data: str = "Belum dinyatakan", unit_confirmations: dict | None = None) -> OasisDataset:
    if isinstance(file, (bytes, bytearray)):
        content = bytes(file)
    elif isinstance(file, str):
        with open(file, "rb") as f:
            content = f.read()
    else:
        content = file.read()
    sha = hashlib.sha256(content).hexdigest()
    xls = pd.ExcelFile(io.BytesIO(content), engine="openpyxl")

    issues: list[Issue] = []
    kamus, meta, dict_issues = read_dictionary(xls)
    issues += [Issue(**d) for d in dict_issues]
    concepts = ConceptConfig.load()
    catalog = build_catalog(kamus, concepts)

    expected_sheets = list(dict.fromkeys(kamus["sheet"].tolist()))
    wb_sheets = {norm_key(s): s for s in xls.sheet_names}

    wide: dict[str, pd.DataFrame] = {}
    mapping_rows: list[dict] = []
    summary_rows: list[dict] = []
    flags: dict[str, list[dict]] = {}

    def add_flag(iid, kind, text):
        flags.setdefault(iid, []).append({"jenis": kind, "teks": text})

    for sheet in expected_sheets:
        cat_s = catalog[catalog["sheet"] == sheet]
        actual = wb_sheets.get(norm_key(sheet))
        srow = {"sheet": sheet, "ditemukan": actual is not None, "field_kamus": len(cat_s)}
        if actual is None:
            issues.append(Issue("Kritis", sheet, "", "Keberadaan sheet",
                                f"Sheet '{sheet}' tercantum di kamus tetapi tidak ada di workbook.",
                                "Unggah workbook yang memuat sheet tersebut."))
            summary_rows.append(srow)
            continue
        df, rinfo = _read_data_sheet(xls, actual)
        if "error" in rinfo:
            issues.append(Issue("Kritis", sheet, "Periode", "Field Periode", rinfo["error"],
                                "Pastikan kolom 'Periode' ada pada baris header."))
            summary_rows.append({**srow, **rinfo})
            continue
        if rinfo.get("baris_kosong_dibuang", 0) or rinfo.get("kolom_kosong_dibuang", 0):
            issues.append(Issue("Info", sheet, "", "Baris/kolom kosong",
                                f"{rinfo['baris_kosong_dibuang']} baris kosong dan {rinfo['kolom_kosong_dibuang']} "
                                f"kolom kosong diabaikan (rentang terpakai lembar kerja lebih besar dari data).",
                                "Tidak berdampak pada analisis; rapikan sheet sumber bila perlu."))
        if rinfo.get("kolom_duplikat"):
            issues.append(Issue("Peringatan", sheet, ", ".join(rinfo["kolom_duplikat"]), "Header duplikat",
                                "Header kolom duplikat; hanya kemunculan pertama yang dipakai.",
                                "Perbaiki header sumber."))

        per_col = next(c for c in df.columns if norm_key(c) == "periode")
        periode, invalid, adjusted = parse_periode_series(df[per_col])
        if invalid:
            issues.append(Issue("Kritis", sheet, "Periode", "Parsing Periode",
                                f"{len(invalid)} nilai Periode tidak valid, contoh: {invalid[:3]}. Baris dibuang.",
                                "Gunakan format YYYYMMDD (akhir bulan)."))
        if adjusted:
            issues.append(Issue("Info", sheet, "Periode", "Akhir bulan",
                                f"{len(adjusted)} tanggal bukan akhir bulan dan disesuaikan, contoh: {adjusted[:2]}.",
                                "Pastikan Periode menggunakan tanggal akhir bulan."))
        df = df.assign(_periode=periode)
        df = df[df["_periode"].notna()]
        was_sorted = df["_periode"].is_monotonic_increasing
        if not was_sorted:
            issues.append(Issue("Peringatan", sheet, "Periode", "Urutan waktu",
                                "Baris tidak terurut menurut Periode; OASIS mengurutkan ulang.",
                                "Urutkan sheet sumber menurut Periode."))
        dup_mask = df["_periode"].duplicated(keep="first")
        n_dup = int(dup_mask.sum())
        if n_dup:
            dups = df.loc[dup_mask, "_periode"].dt.strftime("%Y-%m").unique().tolist()
            issues.append(Issue("Kritis", sheet, "Periode", "Periode duplikat",
                                f"{n_dup} baris memiliki Periode duplikat ({', '.join(dups[:6])}); kemunculan pertama dipakai.",
                                "Hapus duplikat atau tentukan baris yang benar di sumber."))
        df = df[~dup_mask].sort_values("_periode").set_index("_periode")

        full = pd.date_range(df.index.min(), df.index.max(), freq="ME")
        missing_months = full.difference(df.index)
        if len(missing_months):
            issues.append(Issue("Peringatan", sheet, "Periode", "Bulan hilang",
                                f"{len(missing_months)} bulan tidak ada: "
                                f"{', '.join(missing_months.strftime('%Y-%m')[:8])}{'...' if len(missing_months) > 8 else ''}.",
                                "Perhitungan MoM/YoY dan proyeksi memakai periode yang benar-benar tersedia."))

        data_cols = [c for c in df.columns if norm_key(c) != "periode"]
        kfields = cat_s[["indicator_id", "field_asli_kamus", "label_tampil", "unit_raw", "unit_status"]].to_dict("records")
        rec = reconcile_fields(sheet, kfields, data_cols)
        out = pd.DataFrame(index=df.index)
        for m in rec:
            mapping_rows.append({"sheet": sheet, **{k: m.get(k) for k in
                                 ["indicator_id", "field_asli_kamus", "label_tampil", "kolom_sheet", "metode", "keyakinan"]}})
            iid, col = m.get("indicator_id"), m.get("kolom_sheet")
            if iid is None:
                issues.append(Issue("Peringatan", sheet, col, "Kolom di luar kamus",
                                    f"Kolom '{col}' ada di sheet tetapi tidak ada di kamus; tidak dimasukkan ke katalog.",
                                    "Tambahkan ke Kamus_Data bila kolom ini sah."))
                continue
            if col is None:
                issues.append(Issue("Kritis", sheet, m["field_asli_kamus"], "Field kamus",
                                    f"Field kamus '{m['field_asli_kamus']}' tidak ditemukan di sheet.",
                                    "Periksa nama header atau kamus."))
                add_flag(iid, "tidak_tersedia", "Field tidak ditemukan di sheet.")
                out[iid] = np.nan
                continue
            if m["metode"] != "cocok persis":
                lvl = "Peringatan" if "posisi" in m["metode"] else "Info"
                issues.append(Issue(lvl, sheet, m["field_asli_kamus"], "Nama field kamus vs header sheet",
                                    f"Nama di kamus '{m['field_asli_kamus']}' ≠ header sheet '{col}'; dipetakan dengan "
                                    f"metode: {m['metode']} (keyakinan {m['keyakinan']}).",
                                    "Selaraskan nama field kamus dengan header sumber."))
                if "posisi" in m["metode"]:
                    add_flag(iid, "pemetaan_posisi", f"Dipetakan ke kolom '{col}' berdasarkan posisi; perlu konfirmasi.")
            hdr_unit = re.findall(r"\(([^)]*)\)", col)
            if hdr_unit and m.get("unit_status") == "belum_terkonfirmasi":
                issues.append(Issue("Peringatan", sheet, m["field_asli_kamus"], "Konflik satuan kamus vs header",
                                    f"Header sheet mencantumkan '({hdr_unit[-1]})', sedangkan kamus menyatakan satuan "
                                    f"tidak tercantum. OASIS mengikuti kamus: 'satuan belum terkonfirmasi'.",
                                    "Konfirmasi satuan melalui panel Konfirmasi Satuan bila dasar resminya tersedia."))
                add_flag(iid, "konflik_satuan", f"Header sheet: '({hdr_unit[-1]})'; kamus: tidak tercantum.")
            raw_col = df[col]
            num = pd.to_numeric(raw_col, errors="coerce")
            non_numeric = int((raw_col.notna() & num.isna()).sum())
            if non_numeric:
                issues.append(Issue("Peringatan", sheet, m["field_asli_kamus"], "Tipe numerik",
                                    f"{non_numeric} nilai bukan angka diubah menjadi kosong.", "Perbaiki nilai sumber."))
            out[iid] = num.astype(float)

        # pemeriksaan per indikator
        for iid in out.columns:
            info = catalog.loc[catalog["indicator_id"] == iid].iloc[0]
            s = out[iid]
            n_null = int(s.isna().sum())
            if n_null and n_null < len(s):
                null_periods = s[s.isna()].index.strftime("%Y-%m").tolist()
                issues.append(Issue("Peringatan", sheet, info["label_tampil"], "Nilai hilang",
                                    f"{n_null} nilai kosong: {', '.join(null_periods[:6])}{'...' if n_null > 6 else ''}.",
                                    "Statistik dihitung dari observasi yang tersedia; cakupan ditampilkan."))
                add_flag(iid, "nilai_hilang", f"{n_null} nilai kosong.")
            sv = s.dropna()
            if len(sv) and sv.nunique() <= 1:
                issues.append(Issue("Peringatan", sheet, info["label_tampil"], "Seri konstan",
                                    f"Seluruh {len(sv)} observasi bernilai sama ({sv.iloc[0]:g}).",
                                    "Periksa apakah data terisi benar; korelasi/proyeksi tidak bermakna."))
                add_flag(iid, "konstan", "Seri konstan.")
            if len(sv):
                first, last = sv.index.min(), sv.index.max()
                try:
                    mk = pd.Timestamp(info["mulai_kamus"]) if info["mulai_kamus"] else None
                    ak = pd.Timestamp(info["akhir_kamus"]) if info["akhir_kamus"] else None
                except ValueError:
                    mk = ak = None
                if (mk is not None and first > mk) or (ak is not None and last < ak):
                    issues.append(Issue("Peringatan", sheet, info["label_tampil"], "Cakupan vs kamus",
                                        f"Data tersedia {first:%Y-%m}–{last:%Y-%m}, kamus menyatakan "
                                        f"{info['mulai_kamus']}–{info['akhir_kamus']}.", "Periksa kelengkapan data."))
                ytd = detect_ytd_pattern(sv)
                if ytd:
                    add_flag(iid, "indikasi_ytd",
                             f"Pola reset Januari & naik sepanjang tahun pada {ytd['tahun_cocok']} dari "
                             f"{ytd['tahun_uji']} tahun: indikasi kumulatif tahun berjalan (heuristik, belum terkonfirmasi).")
                    issues.append(Issue("Info", sheet, info["label_tampil"], "Pola kumulatif (heuristik)",
                                        flags[iid][-1]["teks"],
                                        "MoM pada data kumulatif menyesatkan; gunakan YoY bulan yang sama atau "
                                        "konfirmasi basis data."))
            if not info["measure_certain"]:
                add_flag(iid, "jenis_belum_pasti", info["measure_note"])
            if info["unit_status"] == "belum_terkonfirmasi":
                add_flag(iid, "satuan_belum_terkonfirmasi", "Satuan belum terkonfirmasi.")
        wide[sheet] = out

        # metadata vs observasi
        if meta is not None:
            mrow = meta[meta["sheet"].map(norm_key) == norm_key(sheet)]
            if not mrow.empty:
                note = collapse_ws(mrow.iloc[0].get("catatan_kualitas"))
                if "kosong" in note.casefold() and int(out.isna().sum().sum()) == 0:
                    issues.append(Issue("Peringatan", sheet, "", "Catatan metadata vs observasi",
                                        f"Metadata_Sheet mencatat: \"{note}\" — namun observasi tidak menemukan nilai "
                                        f"kosong. Nilai kemungkinan telah diisi/diimputasi setelah metadata disusun.",
                                        "Konfirmasi asal nilai pada periode tersebut sebelum dipakai untuk kesimpulan."))
                jo = mrow.iloc[0].get("jumlah_observasi")
                if pd.notna(jo) and int(jo) != len(out):
                    issues.append(Issue("Peringatan", sheet, "", "Jumlah observasi",
                                        f"Metadata: {int(jo)} observasi; data: {len(out)}.", "Periksa kelengkapan."))

        srow.update({
            "baris_data": len(out), "periode_awal": out.index.min().strftime("%Y-%m-%d"),
            "periode_akhir": out.index.max().strftime("%Y-%m-%d"), "bulan_hilang": len(missing_months),
            "periode_duplikat": n_dup, "nilai_kosong": int(out.isna().sum().sum()),
            "field_terpetakan": int(sum(1 for m in rec if m.get("indicator_id") and m.get("kolom_sheet"))),
            "pemetaan_persis": int(sum(1 for m in rec if m.get("metode") == "cocok persis")),
            "baris_kosong_dibuang": rinfo.get("baris_kosong_dibuang", 0),
        })
        summary_rows.append(srow)

    extra = [s for k, s in wb_sheets.items() if k not in {norm_key(x) for x in expected_sheets}
             and k not in {"kamus_data", "metadata_sheet", "kamus data", "metadata sheet", "metadata"}]
    for s in extra:
        issues.append(Issue("Info", s, "", "Sheet di luar kamus",
                            f"Sheet '{s}' tidak tercantum di kamus; diabaikan.", "Tambahkan ke kamus bila relevan."))

    # duplikasi seri lintas sheet
    all_series = [(sh, iid, df[iid].dropna()) for sh, df in wide.items() for iid in df.columns]
    for i in range(len(all_series)):
        for j in range(i + 1, len(all_series)):
            sh1, a, s1 = all_series[i]
            sh2, b, s2 = all_series[j]
            if sh1 == sh2:
                continue
            common = s1.index.intersection(s2.index)
            if len(common) >= 12 and s1[common].nunique() > 1 and np.allclose(s1[common].values, s2[common].values,
                                                                              rtol=1e-9, atol=1e-9):
                la, lb = catalog.set_index("indicator_id").loc[[a, b], "label_tampil"].tolist()
                txt = (f"'{la}' ({sh1}) identik dengan '{lb}' ({sh2}) pada {len(common)} periode bersama.")
                issues.append(Issue("Peringatan", f"{sh1} / {sh2}", f"{la} / {lb}", "Seri identik lintas sheet", txt,
                                    "Kemungkinan salah salin; konfirmasi ke pemilik data."))
                add_flag(a, "identik_lintas_sheet", txt)
                add_flag(b, "identik_lintas_sheet", txt)

    rules = run_relation_rules(catalog, wide)
    for _, r in rules.iterrows():
        if r["status"] in ("gagal", "sebagian"):
            issues.append(Issue("Peringatan", r["sheet"], r["target"], "Uji aturan eksplisit",
                                f"Aturan '{r['id']}' ({r['jenis']}) {r['status']}: lolos {r['lolos_pct']:.0f}% dari "
                                f"{r['n']} periode. {r['detail']}",
                                "Kemungkinan perbedaan skala/satuan atau definisi; jangan bandingkan besaran sebelum "
                                "dikonfirmasi."))
            for iid in [r["target_id"], *r["referensi_id"]]:
                if iid:
                    add_flag(iid, "aturan_gagal", f"Aturan '{r['id']}' {r['status']}. {r['detail']}")

    long_parts = []
    for sh, df in wide.items():
        if df.empty:
            continue
        m = df.reset_index().melt(id_vars="_periode", var_name="indicator_id", value_name="nilai")
        long_parts.append(m.rename(columns={"_periode": "periode"}))
    long = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame(
        columns=["periode", "indicator_id", "nilai"])

    provenance = {
        "nama_file": filename,
        "sha256": sha,
        "waktu_unggah": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S WIB"),
        "status_data": status_data,
        "versi_oasis": __version__,
        "sheet_kamus": "Kamus_Data",
    }
    issues = _consolidate_issues(issues)
    sev_order = {"Kritis": 0, "Peringatan": 1, "Info": 2}
    issues.sort(key=lambda i: (sev_order.get(i.tingkat, 9), i.sheet))
    ds = OasisDataset(catalog=catalog, long=long, wide=wide, mapping=pd.DataFrame(mapping_rows),
                      sheet_summary=pd.DataFrame(summary_rows), issues=issues, rule_results=rules, flags=flags,
                      metadata_sheet=meta, provenance=provenance, base_catalog=catalog)
    if unit_confirmations:
        ds = ds.with_unit_confirmations(unit_confirmations)
    return ds
