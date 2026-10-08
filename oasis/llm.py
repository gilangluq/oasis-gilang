"""Opsional: tafsir pertanyaan dengan LLM (Anthropic Claude).

Aktif hanya bila paket ``anthropic`` terpasang DAN variabel lingkungan
``ANTHROPIC_API_KEY`` tersedia. LLM HANYA menafsirkan pertanyaan menjadi struktur
(indikator, intent, periode, horizon). Yang dikirim: teks pertanyaan + daftar
id/label katalog — TIDAK ADA nilai data. Seluruh perhitungan tetap dilakukan
kode deterministik. Keluaran LLM divalidasi terhadap katalog; id yang tidak ada
dibuang, dan bila gagal parser aturan dipakai.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

from .nlu import INTENT_PATTERNS, ParsedQuery, QuestionParser

DEFAULT_MODEL = os.environ.get("OASIS_LLM_MODEL", "claude-sonnet-5-5")


def available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def interpret(question: str, catalog: pd.DataFrame, parser: QuestionParser,
              data_end: pd.Timestamp | None) -> ParsedQuery:
    base = parser.parse(question, data_end)
    if not available():
        base.notes.append("LLM tidak aktif; dipakai parser aturan.")
        return base
    import anthropic

    items = "\n".join(f"{r.indicator_id} | {r.label_tampil} | {r.sheet}" for r in catalog.itertuples())
    system = (
        "Anda menafsirkan pertanyaan analitik berbahasa Indonesia tentang sektor jasa keuangan menjadi JSON. "
        "Gunakan HANYA indicator_id dari katalog. Jika ambigu, isi 'ambigu' dengan daftar kandidat dan jangan memilih. "
        f"Intent yang sah: {', '.join(INTENT_PATTERNS)}. Format: "
        '{"indikator": [...], "ambigu": [...], "intents": [...], "periode_mulai": "YYYY-MM" | null, '
        '"periode_akhir": "YYYY-MM" | null, "horizon": 3|6|12|null}. Balas JSON saja.'
    )
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(model=DEFAULT_MODEL, max_tokens=600, system=system,
                                     messages=[{"role": "user", "content": f"Katalog:\n{items}\n\nPertanyaan: {question}"}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception as e:  # noqa: BLE001 - fallback aman
        base.notes.append(f"LLM gagal ({type(e).__name__}); dipakai parser aturan.")
        return base
    valid = set(catalog["indicator_id"])
    ids = [i for i in data.get("indikator", []) if i in valid]
    amb = [i for i in data.get("ambigu", []) if i in valid]
    pq = ParsedQuery(question=question, normalized=base.normalized, source=f"LLM ({DEFAULT_MODEL}) + validasi katalog")
    pq.intents = [i for i in data.get("intents", []) if i in INTENT_PATTERNS] or base.intents
    pq.resolved = ids
    if amb:
        pq.ambiguities.append({"tipe": "konsep", "teks": question, "pesan": "LLM menilai pertanyaan ambigu. Pilih:",
                               "pilihan": amb})
    for key, attr, fn in (("periode_mulai", "start", lambda v: pd.Timestamp(v + "-01")),
                          ("periode_akhir", "end", lambda v: pd.Timestamp(v + "-01") + pd.offsets.MonthEnd(0))):
        v = data.get(key)
        try:
            setattr(pq, attr, fn(v) if v else getattr(base, attr))
        except Exception:  # noqa: BLE001
            setattr(pq, attr, getattr(base, attr))
    pq.period_text = base.period_text or ("LLM" if pq.start is not None or pq.end is not None else None)
    h = data.get("horizon")
    pq.horizon = h if h in (3, 6, 12) else base.horizon
    pq.trace = ["Ditafsirkan LLM; id indikator divalidasi terhadap katalog."] + base.trace
    if not ids and not amb:
        base.notes.append("LLM tidak menghasilkan indikator valid; dipakai parser aturan.")
        return base
    return pq
