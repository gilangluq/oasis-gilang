import pandas as pd
import pytest

from oasis.data_prep import detect_ytd_pattern, parse_periode_value, reconcile_fields


@pytest.mark.parametrize("v,expected", [
    (20210131, "2021-01-31"), (20240229, "2024-02-29"), ("20230630", "2023-06-30"), (20230615.0, "2023-06-30"),
    (202312, "2023-12-31"), (pd.Timestamp("2022-05-10"), "2022-05-31"),
])
def test_parse_periode_month_end(v, expected):
    ts, _ = parse_periode_value(v)
    assert ts == pd.Timestamp(expected)


def test_parse_periode_invalid():
    ts, note = parse_periode_value(20231340)
    assert ts is None and "valid" in note


def test_reconcile_methods():
    fields = [{"indicator_id": "a", "field_asli_kamus": "Total Kredit", "label_tampil": "Total Kredit"},
              {"indicator_id": "b", "field_asli_kamus": "Rasio NPL (Gemini)", "label_tampil": "Rasio NPL"},
              {"indicator_id": "c", "field_asli_kamus": "Tagihan Premi X", "label_tampil": "Tagihan Premi X"}]
    cols = ["Total Kredit (Rp M)", "Rasio NPL", "Premi (Rp M)"]
    res = {r["indicator_id"]: r for r in reconcile_fields("BPR", fields, cols)}
    assert res["a"]["metode"] == "cocok setelah normalisasi"
    assert res["b"]["metode"] == "cocok setelah normalisasi"
    assert res["c"]["metode"].startswith("posisi")


def test_ytd_detection():
    idx = pd.date_range("2021-01-31", periods=48, freq="ME")
    ytd = pd.Series([(i % 12 + 1) * 10.0 for i in range(48)], index=idx)
    assert detect_ytd_pattern(ytd) is not None
    stock = pd.Series(range(100, 148), index=idx, dtype=float)
    assert detect_ytd_pattern(stock) is None


def test_demo_quality(demo_ds):
    s = demo_ds.sheet_summary
    assert s["ditemukan"].all() and len(s) == 8
    assert set(demo_ds.catalog["sheet"]) == set(s["sheet"])
    iss = demo_ds.issues_df()
    assert ((iss["pemeriksaan"] == "Nilai hilang") & (iss["sheet"] == "BPR")).any()
    assert (demo_ds.rule_results.query("jenis != 'definisi_tak_teruji'")["status"] == "lolos").all()


def test_no_fields_outside_dictionary(demo_ds):
    # setiap indikator berasal dari Kamus_Data
    assert demo_ds.catalog["field_asli_kamus"].notna().all()
    assert len(demo_ds.catalog) == 89


def test_real_workbook(real_ds):
    iss = real_ds.issues_df()
    assert not (iss["tingkat"] == "Kritis").any()
    assert len(real_ds.catalog) == 89 and real_ds.sheet_summary["ditemukan"].all()
    # seluruh aturan yang dapat diuji dilaporkan dengan status eksplisit
    assert set(real_ds.rule_results["status"]) <= {"lolos", "sebagian", "gagal", "tidak dapat diuji"}
