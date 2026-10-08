"""Pemahaman pertanyaan berbahasa Indonesia berbasis aturan (tanpa LLM).

Langkah parser (seluruhnya deterministik & dapat diuji):
1. Normalisasi teks.
2. Ekstraksi periode (rentang, bulan, tahun, kuartal, semester, N bulan/tahun
   terakhir, sejak, tahun berjalan) dan horizon proyeksi.
3. Pencocokan konsep indikator (sinonim + label kamus, frasa terpanjang dulu),
   lalu subsektor.
4. Deteksi intent dari sisa teks.
5. Resolusi ke indicator_id katalog. Bila lebih dari satu kandidat dan
   pertanyaan tidak meminta perbandingan, kembalikan AMBIGUITAS (tidak memilih
   diam-diam).
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import pandas as pd

from .catalog import ConceptConfig

MONTHS = {
    "januari": 1, "jan": 1, "january": 1,
    "februari": 2, "pebruari": 2, "feb": 2, "february": 2,
    "maret": 3, "mar": 3, "march": 3,
    "april": 4, "apr": 4,
    "mei": 5, "may": 5,
    "juni": 6, "jun": 6, "june": 6,
    "juli": 7, "jul": 7, "july": 7,
    "agustus": 8, "agu": 8, "agt": 8, "aug": 8, "august": 8,
    "september": 9, "sep": 9, "sept": 9,
    "oktober": 10, "okt": 10, "oct": 10, "october": 10,
    "november": 11, "nov": 11, "nop": 11,
    "desember": 12, "des": 12, "dec": 12, "december": 12,
}
_M = "|".join(sorted(MONTHS, key=len, reverse=True))
_Y = r"(20\d{2}|19\d{2})"
_SEP = r"(?:sampai dengan|sampai|hingga|s/d|s\.d\.?|sd|ke|-|–|—|to)"
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}

INTENT_PATTERNS: dict[str, str] = {
    "tertinggi": r"tertinggi|maksimum|maksimal|terbesar|puncak|paling tinggi|\bmax\b|highest",
    "terendah": r"terendah|minimum|minimal|terkecil|paling rendah|\bmin\b|lowest|titik terendah",
    "rata_rata": r"rata-rata|rata rata|rerata|average|\bmean\b|rataan",
    "median": r"\bmedian\b|nilai tengah",
    "std": r"simpangan baku|standar deviasi|deviasi standar|\bstd\b|stdev|volatilitas|variabilitas",
    "kuartil": r"kuartil|quartile|persentil|\biqr\b|sebaran|distribusi|statistik deskriptif|statistik",
    "mom": r"\bmom\b|month on month|month-on-month|bulan ke bulan|bulan lalu|bulan sebelumnya|perubahan bulanan",
    "yoy": r"\byoy\b|year on year|year-on-year|tahun lalu|tahun sebelumnya|secara tahunan|perubahan tahunan|pertumbuhan tahunan",
    "tren": r"\btren\b|trend|perkembangan|pergerakan|kecenderungan|naik atau turun|arah",
    "bandingkan": r"bandingkan|perbandingan|dibandingkan|\bbanding\b|\bvs\b|versus|compare|antara",
    "korelasi": r"korelasi|hubungan|keterkaitan|kaitan|asosiasi|scatter|bergerak bersama|co-?movement",
    "proyeksi": r"proyeksi|proyeksikan|forecast|prediksi|ramal|perkiraan|ke depan|kedepan|mendatang",
    "perhatian": r"perhatian|anomali|tidak wajar|outlier|janggal|lonjakan|red flag|early warning|peringatan dini|waspada",
    "total": r"\btotal\b|\bjumlah\b|jumlahkan|akumulasi|dijumlah",
    "terakhir": r"terakhir|terkini|saat ini|sekarang|\bposisi\b|latest|terbaru",
    "ringkasan": r"ringkasan|ringkas|rangkum|summary|gambaran|kondisi|profil|ikhtisar",
}
STAT_INTENTS = ["tertinggi", "terendah", "rata_rata", "median", "std", "kuartil", "mom", "yoy", "tren", "total",
                "terakhir"]


@dataclass
class ConceptMatch:
    teks: str
    concept_ids: list[str]
    pos: int


@dataclass
class ParsedQuery:
    question: str
    normalized: str
    intents: list[str] = field(default_factory=list)
    subsectors: list[str] = field(default_factory=list)
    concept_matches: list[ConceptMatch] = field(default_factory=list)
    resolved: list[str] = field(default_factory=list)
    ambiguities: list[dict] = field(default_factory=list)
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    period_text: str | None = None
    horizon: int | None = None
    notes: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    source: str = "parser aturan"

    @property
    def is_ambiguous(self) -> bool:
        return bool(self.ambiguities)

    def to_dict(self) -> dict:
        return {
            "intents": self.intents, "subsektor": self.subsectors,
            "konsep": [{"teks": c.teks, "konsep": c.concept_ids} for c in self.concept_matches],
            "indikator": self.resolved, "ambiguitas": self.ambiguities,
            "periode_mulai": None if self.start is None else self.start.strftime("%Y-%m-%d"),
            "periode_akhir": None if self.end is None else self.end.strftime("%Y-%m-%d"),
            "teks_periode": self.period_text, "horizon": self.horizon, "catatan": self.notes,
            "sumber_tafsir": self.source,
        }


def normalize(text: str) -> str:
    t = text.casefold()
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"[?!,;:\"()\[\]]", " ", t)
    t = re.sub(r"\.(?!\d)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _mask(text: str, span: tuple[int, int]) -> str:
    a, b = span
    return text[:a] + " " * (b - a) + text[b:]


def _month_end(y: int, m: int) -> pd.Timestamp:
    return (pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)).normalize()


def _month_start(y: int, m: int) -> pd.Timestamp:
    return pd.Timestamp(year=y, month=m, day=1)


# ---------------------------------------------------------------------------
# Periode & horizon
# ---------------------------------------------------------------------------
def extract_horizon(t: str) -> tuple[int | None, str, str | None]:
    pats = [
        r"(\d{1,2})\s*bulan\s*(?:ke\s*depan|kedepan|mendatang|berikutnya|lagi)",
        r"(?:proyeksi\w*|forecast\w*|prediksi\w*|ramal\w*|perkiraan)\s+(?:\S+\s+){0,5}?(\d{1,2})\s*bulan",
        r"(?:horizon|horison)\s*(\d{1,2})",
    ]
    for p in pats:
        m = re.search(p, t)
        if m:
            # hanya bagian angka+"bulan" yang dimask agar kata indikator & kata kunci proyeksi tetap terbaca
            a = m.start(1)
            b = m.end()
            if not re.search(r"bulan", t[a:b]):
                b = m.end(1)
            return int(m.group(1)), _mask(t, (a, b)), t[a:b]
    m = re.search(r"(setahun|satu tahun|1 tahun)\s*(ke\s*depan|kedepan|mendatang)", t)
    if m:
        return 12, _mask(t, m.span()), m.group(0)
    m = re.search(r"(enam|setengah tahun|semester)\s*(bulan)?\s*(ke\s*depan|kedepan|mendatang)", t)
    if m:
        return 6, _mask(t, m.span()), m.group(0)
    m = re.search(r"(tiga|satu kuartal|sekuartal|triwulan)\s*(bulan)?\s*(ke\s*depan|kedepan|mendatang)", t)
    if m:
        return 3, _mask(t, m.span()), m.group(0)
    return None, t, None


def extract_period(t: str, data_end: pd.Timestamp | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str, str | None]:
    """Kembalikan (mulai, akhir, teks_tersisa, teks_periode)."""
    de = data_end
    # N bulan/tahun terakhir
    m = re.search(r"(\d{1,3})\s*(bulan|tahun)\s*(terakhir|belakangan|ke belakang)", t)
    if m and de is not None:
        n = int(m.group(1))
        months = n if m.group(2) == "bulan" else 12 * n
        start = (de - pd.offsets.MonthEnd(months - 1)).normalize()
        start = pd.Timestamp(year=start.year, month=start.month, day=1)
        return start, de, _mask(t, m.span()), m.group(0)
    m = re.search(r"(setahun|satu tahun)\s*(terakhir|belakangan)", t)
    if m and de is not None:
        start = (de - pd.offsets.MonthEnd(11)).normalize()
        return pd.Timestamp(year=start.year, month=start.month, day=1), de, _mask(t, m.span()), m.group(0)
    m = re.search(r"\b(ytd|tahun berjalan|tahun ini)\b", t)
    if m and de is not None:
        return _month_start(de.year, 1), de, _mask(t, m.span()), m.group(0)
    # rentang bulan-tahun / tahun
    m = re.search(rf"(?:dari\s+|periode\s+|antara\s+)?(?:({_M})\s+)?{_Y}\s*{_SEP}\s*(?:({_M})\s+)?{_Y}", t)
    if m:
        m1, y1, m2, y2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        start = _month_start(y1, MONTHS[m1] if m1 else 1)
        end = _month_end(y2, MONTHS[m2] if m2 else 12)
        return start, end, _mask(t, m.span()), m.group(0)
    m = re.search(rf"({_M})\s*{_SEP}\s*({_M})\s+{_Y}", t)
    if m:
        y = int(m.group(3))
        return (_month_start(y, MONTHS[m.group(1)]), _month_end(y, MONTHS[m.group(2)]), _mask(t, m.span()),
                m.group(0))
    m = re.search(rf"(?:sejak|mulai|semenjak)\s+(?:({_M})\s+)?{_Y}", t)
    if m:
        start = _month_start(int(m.group(2)), MONTHS[m.group(1)] if m.group(1) else 1)
        return start, de, _mask(t, m.span()), m.group(0)
    m = re.search(rf"(?:sebelum|hingga|sampai)\s+(?:({_M})\s+)?{_Y}", t)
    if m:
        end = _month_end(int(m.group(2)), MONTHS[m.group(1)] if m.group(1) else 12)
        return None, end, _mask(t, m.span()), m.group(0)
    m = re.search(rf"(?:q|kuartal\s*|triwulan\s*|tw\s*)([1-4]|iv|iii|ii|i)\s*(?:tahun\s*)?{_Y}", t)
    if m:
        q = ROMAN.get(m.group(1), None) or int(m.group(1))
        y = int(m.group(2))
        return _month_start(y, 3 * q - 2), _month_end(y, 3 * q), _mask(t, m.span()), m.group(0)
    m = re.search(rf"(?:semester\s*|smt\s*|\bs|\bh)([12]|ii|i)\s*(?:tahun\s*)?{_Y}", t)
    if m:
        h = ROMAN.get(m.group(1), None) or int(m.group(1))
        y = int(m.group(2))
        return _month_start(y, 6 * h - 5), _month_end(y, 6 * h), _mask(t, m.span()), m.group(0)
    m = re.search(rf"(?:akhir\s+(?:tahun\s+)?){_Y}", t)
    if m:
        y = int(m.group(1))
        return _month_start(y, 12), _month_end(y, 12), _mask(t, m.span()), m.group(0)
    m = re.search(rf"({_M})\s+{_Y}", t)
    if m:
        y, mo = int(m.group(2)), MONTHS[m.group(1)]
        return _month_start(y, mo), _month_end(y, mo), _mask(t, m.span()), m.group(0)
    m = re.search(rf"(?:tahun\s+)?\b{_Y}\b", t)
    if m:
        y = int(m.group(1))
        return _month_start(y, 1), _month_end(y, 12), _mask(t, m.span()), m.group(0)
    return None, None, t, None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
class QuestionParser:
    def __init__(self, catalog: pd.DataFrame, concepts: ConceptConfig | None = None):
        self.catalog = catalog
        self.cfg = concepts or ConceptConfig.load()
        present = set(catalog["concept_id"])
        phrases: dict[str, set[str]] = {}
        for k in self.cfg.konsep:
            if k["id"] not in present:
                continue
            for s in k.get("sinonim", []) + k.get("labels", []):
                phrases.setdefault(normalize(s), set()).add(k["id"])
        for _, r in catalog.iterrows():
            # hanya label ternormalisasi (tanpa akhiran subsektor) agar subsektor tetap terdeteksi terpisah
            phrases.setdefault(normalize(r["label_norm"]), set()).add(r["concept_id"])
        for s, ids in self.cfg.ambigu.items():
            ids = [i for i in ids if i in present]
            if ids:
                phrases.setdefault(normalize(s), set()).update(ids)
        self.concept_phrases = sorted(phrases.items(), key=lambda kv: len(kv[0]), reverse=True)
        sheets = set(catalog["sheet"])
        sub: list[tuple[str, list[str]]] = []
        for sheet, syns in self.cfg.subsektor.items():
            if sheet in sheets:
                for s in syns + [sheet]:
                    sub.append((normalize(s), [sheet]))
        for g, members in self.cfg.kelompok.items():
            mem = [m for m in members if m in sheets]
            if mem:
                sub.append((normalize(g), mem))
        self.sub_phrases = sorted(sub, key=lambda kv: len(kv[0]), reverse=True)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _find(phrase: str, text: str):
        return re.search(rf"(?<![\w+]){re.escape(phrase)}(?![\w+])", text)

    def _match_concepts(self, t: str) -> tuple[list[ConceptMatch], str]:
        matches = []
        for phrase, ids in self.concept_phrases:
            if not phrase:
                continue
            while True:
                m = self._find(phrase, t)
                if not m:
                    break
                matches.append(ConceptMatch(m.group(0), sorted(ids), m.start()))
                t = _mask(t, m.span())
        return sorted(matches, key=lambda c: c.pos), t

    def _match_subsectors(self, t: str) -> tuple[list[str], list[str], str]:
        found, groups = [], []
        for phrase, sheets in self.sub_phrases:
            m = self._find(phrase, t)
            if m:
                if len(sheets) == 1:
                    found += [s for s in sheets if s not in found]
                else:
                    groups.append(phrase)
                    found += [s for s in sheets if s not in found]
                t = _mask(t, m.span())
        return found, groups, t

    def _fuzzy(self, t: str) -> list[ConceptMatch]:
        words = t.split()
        grams = set()
        for n in (1, 2, 3):
            for i in range(len(words) - n + 1):
                g = " ".join(words[i:i + n])
                if len(g) >= 4:
                    grams.add(g)
        keys = [p for p, _ in self.concept_phrases if len(p) >= 4]
        out = []
        for g in grams:
            close = difflib.get_close_matches(g, keys, n=1, cutoff=0.86)
            if close:
                ids = dict(self.concept_phrases)[close[0]]
                out.append(ConceptMatch(f"{g}→{close[0]}", sorted(ids), t.find(g)))
        uniq = {}
        for c in out:
            uniq.setdefault(tuple(c.concept_ids), c)
        return list(uniq.values())

    def label(self, iid: str) -> str:
        r = self.catalog.loc[self.catalog["indicator_id"] == iid].iloc[0]
        return f"{r['label_tampil']} — {r['sheet']}"

    # -- main -------------------------------------------------------------
    def parse(self, question: str, data_end: pd.Timestamp | None = None) -> ParsedQuery:
        t = normalize(question)
        pq = ParsedQuery(question=question, normalized=t)

        horizon, t, htxt = extract_horizon(t)
        if htxt:
            pq.trace.append(f"Horizon proyeksi terdeteksi: '{htxt}' → {horizon} bulan")
        start, end, t, ptxt = extract_period(t, data_end)
        pq.start, pq.end, pq.period_text = start, end, ptxt
        if ptxt:
            pq.trace.append(f"Periode terdeteksi: '{ptxt}' → {start.date() if start is not None else 'awal data'} s.d. "
                            f"{end.date() if end is not None else 'akhir data'}")

        cms, t2 = self._match_concepts(t)
        if not cms:
            cms = self._fuzzy(t)
            if cms:
                pq.trace.append("Indikator dicocokkan secara fuzzy (kemungkinan salah ketik): "
                                + "; ".join(c.teks for c in cms))
        pq.concept_matches = cms
        for c in cms:
            pq.trace.append(f"Frasa '{c.teks}' → konsep {', '.join(c.concept_ids)}")
        subs, groups, t3 = self._match_subsectors(t2)
        pq.subsectors = subs
        if subs:
            pq.trace.append(f"Subsektor: {', '.join(subs)}" + (f" (kelompok: {', '.join(groups)})" if groups else ""))

        intents = [k for k, p in INTENT_PATTERNS.items() if re.search(p, t3)]
        if horizon and "proyeksi" not in intents:
            intents.append("proyeksi")
        if "proyeksi" in intents:
            if horizon is None:
                horizon = 6
                pq.notes.append("Horizon proyeksi tidak disebut; dipakai 6 bulan (pilihan: 3, 6, 12).")
            elif horizon not in (3, 6, 12):
                new = min((3, 6, 12), key=lambda h: abs(h - horizon))
                pq.notes.append(f"Horizon {horizon} bulan disesuaikan ke {new} bulan (pilihan yang didukung: 3, 6, 12).")
                horizon = new
        pq.horizon = horizon
        if "terakhir" in intents and len(intents) > 1:
            intents.remove("terakhir")
        pq.intents = intents
        if intents:
            pq.trace.append(f"Intent: {', '.join(intents)}")
        self._resolve(pq)
        if not pq.intents and (pq.resolved or pq.ambiguities):
            pq.intents = ["bandingkan"] if len(pq.resolved) > 1 else ["ringkasan"]
            pq.trace.append(f"Intent tidak eksplisit → default '{pq.intents[0]}'")
        return pq

    def _resolve(self, pq: ParsedQuery) -> None:
        cat = self.catalog
        multi_ok = any(i in pq.intents for i in ("bandingkan", "korelasi"))
        resolved: list[str] = []
        if not pq.concept_matches:
            if pq.subsectors and any(i in pq.intents for i in ("perhatian", "ringkasan")):
                pq.resolved = cat[cat["sheet"].isin(pq.subsectors)]["indicator_id"].tolist()
                pq.notes.append(f"Mode pemindaian: seluruh {len(pq.resolved)} indikator pada "
                                f"{', '.join(pq.subsectors)} diperiksa.")
                pq.trace.append("Tanpa indikator spesifik + intent perhatian/ringkasan → pindai semua indikator subsektor")
                return
            if pq.subsectors:
                opts = cat[cat["sheet"].isin(pq.subsectors)]["indicator_id"].tolist()
                pq.ambiguities.append({"tipe": "indikator", "teks": ", ".join(pq.subsectors),
                                       "pesan": "Subsektor dikenali, tetapi indikator belum disebut. Pilih indikator:",
                                       "pilihan": opts})
            else:
                pq.notes.append("Indikator maupun subsektor tidak dikenali dari pertanyaan. Coba sebut nama indikator "
                                "(mis. 'Rasio NPL', 'Total Aset', 'IHSG') dan subsektornya.")
            return
        for cm in pq.concept_matches:
            rows = cat[cat["concept_id"].isin(cm.concept_ids)]
            cand = rows
            if pq.subsectors:
                in_sub = rows[rows["sheet"].isin(pq.subsectors)]
                if in_sub.empty:
                    if len(set(rows["sheet"])) == 1:
                        cand = rows  # konsep hanya ada di satu subsektor (mis. IHSG) -> pakai
                    else:
                        pq.notes.append(
                            f"'{cm.teks}' tidak tersedia pada subsektor {', '.join(pq.subsectors)}; tersedia di: "
                            f"{', '.join(sorted(set(rows['sheet'])))}.")
                        cand = rows
                        pq.ambiguities.append({"tipe": "subsektor", "teks": cm.teks,
                                               "pesan": f"'{cm.teks}' tidak ada di subsektor yang disebut. Pilih:",
                                               "pilihan": cand["indicator_id"].tolist()})
                        continue
                else:
                    cand = in_sub
            ids = cand["indicator_id"].tolist()
            if len(cm.concept_ids) > 1 and len(set(cand["concept_id"])) > 1:
                pq.ambiguities.append({"tipe": "konsep", "teks": cm.teks,
                                       "pesan": f"Frasa '{cm.teks}' cocok dengan beberapa indikator. Pilih yang dimaksud:",
                                       "pilihan": ids})
                continue
            if len(ids) == 1:
                resolved += ids
            elif len(ids) > 1:
                explicit = pq.subsectors and len(pq.subsectors) >= 2 and set(cand["sheet"]) <= set(pq.subsectors)
                if multi_ok or explicit:
                    resolved += ids
                    if not explicit:
                        pq.notes.append(f"'{cm.teks}' tersedia di {len(ids)} subsektor; seluruhnya dibandingkan.")
                else:
                    pq.ambiguities.append({"tipe": "subsektor", "teks": cm.teks,
                                           "pesan": f"'{cm.teks}' tersedia di beberapa subsektor. Pilih subsektor:",
                                           "pilihan": ids})
        pq.resolved = list(dict.fromkeys(resolved))
