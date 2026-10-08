import pandas as pd

END = pd.Timestamp("2026-06-30")


def test_compare_two_subsectors(parser):
    pq = parser.parse("Bandingkan tren Rasio NPL Bank Umum dan BPR 2023-2025", END)
    assert set(pq.resolved) == {"bank_umum::rasio_npl", "bpr::rasio_npl"}
    assert pq.start == pd.Timestamp("2023-01-01") and pq.end == pd.Timestamp("2025-12-31")
    assert {"tren", "bandingkan"} <= set(pq.intents)


def test_ambiguous_no_silent_choice(parser):
    pq = parser.parse("Berapa NPL tertinggi?", END)
    assert pq.resolved == [] and pq.is_ambiguous
    assert set(pq.ambiguities[0]["pilihan"]) == {"bank_umum::rasio_npl", "bpr::rasio_npl"}


def test_ambiguous_concept(parser):
    pq = parser.parse("klaim asuransi jiwa", END)
    assert pq.is_ambiguous and pq.ambiguities[0]["tipe"] == "konsep"


def test_cross_sector(parser):
    pq = parser.parse("Bagaimana hubungan IHSG dan total investasi dana pensiun?", END)
    assert pq.resolved == ["pasar_modal::ihsg", "dana_pensiun::total_investasi"]
    assert "korelasi" in pq.intents


def test_relative_period_and_horizon(parser):
    pq = parser.parse("Proyeksi total aset BPR 12 bulan ke depan", END)
    assert pq.horizon == 12 and pq.resolved == ["bpr::total_aset_rp_m"] and pq.start is None
    pq2 = parser.parse("NPF pembiayaan 12 bulan terakhir", END)
    assert pq2.start == pd.Timestamp("2025-07-01") and pq2.horizon is None


def test_horizon_snapped(parser):
    pq = parser.parse("ramalkan IHSG 9 bulan ke depan", END)
    assert pq.horizon == 6 and any("disesuaikan" in n for n in pq.notes)


def test_quarter_and_month(parser):
    pq = parser.parse("kuartil BOPO bank umum q2 2025", END)
    assert (pq.start, pq.end) == (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-30"))
    pq = parser.parse("laba bank umum juni 2026", END)
    assert (pq.start, pq.end) == (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-30"))


def test_concept_not_in_subsector(parser):
    pq = parser.parse("NPL asuransi umum", END)
    assert pq.is_ambiguous and any("tidak tersedia" in n for n in pq.notes)


def test_typo_fuzzy(parser):
    pq = parser.parse("rasoi npf pembiayaan", END)
    assert pq.resolved == ["perusahaan_pembiayaan::rasio_npf"]


def test_scan_mode(parser):
    pq = parser.parse("Apa yang perlu perhatian di bank umum?", END)
    assert len(pq.resolved) == 13
