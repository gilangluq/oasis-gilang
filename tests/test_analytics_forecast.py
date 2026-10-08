import numpy as np
import pandas as pd

from oasis import analytics as A
from oasis import engine
from oasis.forecast import forecast
from oasis.nlu import ParsedQuery

END = pd.Timestamp("2026-06-30")


def test_changes_pp_vs_pct(demo_ds):
    npl = demo_ds.info("bank_umum::rasio_npl")
    s = demo_ds.series("bank_umum::rasio_npl")
    ch = A.monthly_changes(s, npl)
    assert ch.attrs["mode"] == "pp"
    assert np.isclose(ch["mom"].iloc[-1], s.iloc[-1] - s.iloc[-2])
    kr = demo_ds.info("bank_umum::total_kredit_rp_m")
    s2 = demo_ds.series("bank_umum::total_kredit_rp_m")
    ch2 = A.monthly_changes(s2, kr)
    assert ch2.attrs["mode"] == "pct"
    assert np.isclose(ch2["yoy"].iloc[-1], (s2.iloc[-1] / s2.iloc[-13] - 1) * 100)


def test_mom_respects_missing_months():
    idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-04-30"])
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    ch = A.monthly_changes(s, {"measure_type": "rasio", "unit": "%", "unit_status": "terkonfirmasi_kamus"})
    assert np.isnan(ch.loc["2024-04-30", "mom"])  # Maret hilang → tidak dibandingkan dengan Februari


def test_stock_not_summed(demo_ds, parser):
    pq = parser.parse("Jumlah kredit bank umum tahun 2024", END)
    res = engine.run(demo_ds, pq)
    assert not any(a["judul"] == "Jumlah selama periode" for a in res.answers)
    assert any("STOK" in w for w in res.warnings)


def test_flow_summed(demo_ds, parser):
    pq = parser.parse("jumlah volume transaksi 2024", END)
    res = engine.run(demo_ds, pq)
    a = [x for x in res.answers if x["judul"] == "Jumlah selama periode"][0]
    s = demo_ds.series("pasar_modal::volume_transaksi_miliar_saham", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
    assert np.isclose(a["nilai"], s.sum()) and len(s) == 12


def test_stats_and_provenance(demo_ds, parser):
    pq = parser.parse("rata-rata dan median ROA perusahaan pembiayaan tahun 2024", END)
    res = engine.run(demo_ds, pq)
    s = demo_ds.series("perusahaan_pembiayaan::rasio_roa", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
    got = {a["judul"]: a["nilai"] for a in res.answers}
    assert np.isclose(got["Rata-rata"], s.mean()) and np.isclose(got["Median"], s.median())
    p = res.provenance.iloc[0]
    assert p["jumlah observasi"] == 12 and p["status data"] == "Simulasi"


def test_window_consistency(demo_ds, parser):
    pq = parser.parse("Bandingkan tren Rasio NPL Bank Umum dan BPR 2023-2025", END)
    res = engine.run(demo_ds, pq, gstart=pd.Timestamp("2024-01-31"), gend=END)
    tbl = res.table()
    assert tbl["periode"].min() == "2024-01-31" and tbl["periode"].max() == "2025-12-31"
    assert res.stats["bank_umum::rasio_npl"]["n"] == 24


def test_anomaly_detected_in_demo(demo_ds):
    res = engine.run(demo_ds, ParsedQuery("x", "x", intents=["perhatian"]), ["bpr::rasio_npl"])
    assert not res.anomalies["bpr::rasio_npl"].empty


def test_forecast_guards():
    idx = pd.date_range("2024-01-31", periods=10, freq="ME")
    fr = forecast(pd.Series(np.arange(10.0), index=idx), 6)
    assert not fr.ok and "kurang" in fr.alasan
    fr = forecast(pd.Series(np.ones(40), index=pd.date_range("2021-01-31", periods=40, freq="ME")), 3)
    assert not fr.ok and "konstan" in fr.alasan
    assert not forecast(pd.Series(np.arange(40.0), index=pd.date_range("2021-01-31", periods=40, freq="ME")), 9).ok


def test_forecast_selects_seasonal():
    idx = pd.date_range("2020-01-31", periods=60, freq="ME")
    y = pd.Series(100 + 10 * np.sin(2 * np.pi * np.arange(60) / 12), index=idx)
    fr = forecast(y, 12)
    assert fr.ok and fr.metode == "snaive" and len(fr.prediksi) == 12
    assert (fr.prediksi["bawah"] <= fr.prediksi["prediksi"]).all()
    assert fr.evaluasi is not None and "MAE" in fr.evaluasi
