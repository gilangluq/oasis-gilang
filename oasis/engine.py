"""Orkestrasi: pertanyaan terurai -> hasil analitik lengkap (satu konteks data).

Semua komponen tampilan (grafik, tabel, kotak perhatian, ringkasan) memakai
objek :class:`AnalysisResult` yang sama sehingga periode dan data konsisten.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import analytics as A
from . import attention as T
from . import compat as C
from .data_prep import OasisDataset
from .forecast import ForecastResult, forecast
from .nlu import ParsedQuery


@dataclass
class AnalysisResult:
    query: ParsedQuery
    ids: list[str]
    infos: dict[str, dict]
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    window_note: str
    series: dict[str, pd.Series]        # dalam jendela periode
    series_full: dict[str, pd.Series]   # seluruh histori (untuk pembanding MoM/YoY)
    changes: dict[str, pd.DataFrame]
    stats: dict[str, dict]
    trends: dict[str, dict]
    answers: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    compat: dict | None = None
    pairs: list[dict] = field(default_factory=list)
    aligned: dict | None = None
    correlations: list[dict] = field(default_factory=list)
    anomalies: dict[str, pd.DataFrame] = field(default_factory=dict)
    anomaly_method: dict[str, str] = field(default_factory=dict)
    thresholds: dict[str, list] = field(default_factory=dict)
    attention: list[dict] = field(default_factory=list)
    forecasts: dict[str, ForecastResult] = field(default_factory=dict)
    chart_options: list[str] = field(default_factory=list)
    default_chart: str = "Garis"
    provenance: pd.DataFrame | None = None

    def label(self, iid: str, with_sheet: bool = True) -> str:
        i = self.infos[iid]
        return f"{i['label_tampil']} — {i['sheet']}" if with_sheet else i["label_tampil"]

    def table(self, mode: str = "nilai") -> pd.DataFrame:
        """Tabel data pendukung (grid) — persis data yang dipakai grafik & ringkasan."""
        frames = []
        for iid in self.ids:
            ch = self.changes[iid]
            if ch.empty:
                continue
            m = ch.attrs.get("mode", "abs")
            u = A.unit_text(self.infos[iid])
            df = pd.DataFrame({
                "periode": ch.index.strftime("%Y-%m-%d"),
                "subsektor": self.infos[iid]["sheet"],
                "indikator": self.infos[iid]["label_tampil"],
                "field_sumber": self.infos[iid]["field_asli_kamus"],
                "nilai": ch["nilai"].values,
                "satuan": u,
                f"MoM": ch["mom"].values,
                f"YoY": ch["yoy"].values,
                "satuan_perubahan": {"pp": "pp", "pct": "%", "abs": f"selisih ({u})"}[m],
                "anomali": [p in set(self.anomalies.get(iid, pd.DataFrame(columns=["periode"]))["periode"])
                            for p in ch.index],
            })
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _window(ds: OasisDataset, pq: ParsedQuery, gstart, gend) -> tuple:
    dmin, dmax = ds.period_bounds()
    start = gstart if gstart is not None else dmin
    end = gend if gend is not None else dmax
    note = f"Filter periode global: {A.period_label(start)}–{A.period_label(end)}."
    if pq.start is not None or pq.end is not None:
        qs = pq.start if pq.start is not None else start
        qe = pq.end if pq.end is not None else end
        ns, ne = max(start, qs), min(end, qe)
        note = (f"Periode dari pertanyaan ('{pq.period_text}') {A.period_label(qs)}–{A.period_label(qe)}, "
                f"diiriskan dengan filter global {A.period_label(start)}–{A.period_label(end)} → "
                f"{A.period_label(ns)}–{A.period_label(ne)}.")
        start, end = ns, ne
    return start, end, note


def run(ds: OasisDataset, pq: ParsedQuery, ids: list[str] | None = None, gstart=None, gend=None,
        threshold_overrides: dict | None = None, horizon: int | None = None) -> AnalysisResult:
    ids = list(dict.fromkeys(ids if ids is not None else pq.resolved))
    infos = {i: ds.info(i) for i in ids}
    start, end, wnote = _window(ds, pq, gstart, gend)
    res = AnalysisResult(query=pq, ids=ids, infos=infos, start=start, end=end, window_note=wnote, series={},
                         series_full={}, changes={}, stats={}, trends={})
    if start is not None and end is not None and start > end:
        res.warnings.append("Periode pertanyaan tidak beririsan dengan filter periode global; tidak ada data.")
    intents = set(pq.intents)
    dmin, dmax = ds.period_bounds()

    for iid in ids:
        info = infos[iid]
        full = ds.series(iid)
        s = full[(full.index >= start) & (full.index <= end)] if start is not None else full
        res.series_full[iid] = full
        res.series[iid] = s
        ch_full = A.monthly_changes(full, info)
        ch = ch_full.loc[ch_full.index.isin(s.index)]
        ch.attrs["mode"] = ch_full.attrs.get("mode")
        res.changes[iid] = ch
        res.stats[iid] = A.describe(s)
        res.trends[iid] = A.trend(s, info)
        flags = ds.flags.get(iid, [])
        if s.empty:
            avail = f"{A.period_label(full.index.min())}–{A.period_label(full.index.max())}" if not full.empty else "-"
            res.warnings.append(f"Tidak ada data {res.label(iid)} pada periode terpilih. Data tersedia: {avail}.")
            res.alternatives.append(f"Ubah periode ke rentang {avail} atau hapus batasan periode pada pertanyaan.")
            continue
        an, meth = T.detect_anomalies(s, info, flags)
        res.anomalies[iid], res.anomaly_method[iid] = an, meth
        ths = T.thresholds_for(info, threshold_overrides)
        hits = T.check_thresholds(s, info, ths)
        res.thresholds[iid] = hits
        res.attention += T.attention_items(s, info, flags, an, hits)
        if info["unit_status"] == "belum_terkonfirmasi":
            res.warnings.append(f"{res.label(iid)}: satuan belum terkonfirmasi — nilai ditampilkan dalam satuan "
                                f"sumber tanpa label satuan.")
        for f in flags:
            if f["jenis"] == "indikasi_ytd":
                res.warnings.append(f"{res.label(iid)}: {f['teks']} MoM pada seri kumulatif tidak mencerminkan "
                                    f"arus bulanan.")
            if f["jenis"] in ("aturan_gagal", "identik_lintas_sheet"):
                res.warnings.append(f"{res.label(iid)}: {f['teks']}")
        _answers_for(res, iid, intents, ds)
        if "proyeksi" in intents:
            h = horizon or pq.horizon or 6
            fr = forecast(s, h, flags, info)
            big = an[an["skor_z"].abs() >= 10] if not an.empty else an
            if fr.ok and not big.empty:
                fr.catatan.append("Terdapat lompatan sangat besar pada data latih ("
                                  + ", ".join(A.period_label(p) for p in big["periode"]) +
                                  "; |z| ≥ 10) — kemungkinan perubahan cakupan/struktural. Metode drift/tren ikut "
                                  "terpengaruh dan interval melebar; pertimbangkan memendekkan periode latih lewat "
                                  "filter periode.")
            res.forecasts[iid] = fr

    res.attention = T.sort_items(res.attention)
    if len(ids) >= 2:
        sers = {i: res.series[i] for i in ids}
        res.compat = C.group_level([infos[i] for i in ids], sers)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                p = C.pair(infos[ids[a]], infos[ids[b]], sers[ids[a]], sers[ids[b]])
                res.pairs.append({"a": ids[a], "b": ids[b], **p})
        res.aligned = A.align(sers)
        miss = {k: v for k, v in res.aligned["hilang"].items() if v}
        if miss:
            for k, v in miss.items():
                res.warnings.append(f"{res.label(k)} tidak memiliki data pada {len(v)} periode yang ada di seri lain "
                                    f"(contoh: {', '.join(A.period_label(p) for p in v[:4])}). Perbandingan memakai "
                                    f"irisan {res.aligned['n_irisan']} periode.")
        if "korelasi" in intents or len(ids) == 2:
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    ia, ib = ids[a], ids[b]
                    c = A.correlation(res.series[ia], res.series[ib], infos[ia], infos[ib],
                                      ds.flags.get(ia, []), ds.flags.get(ib, []))
                    res.correlations.append({"a": ia, "b": ib, **c})
    _chart_options(res)
    res.provenance = provenance_table(ds, res)
    return res


def _answers_for(res: AnalysisResult, iid: str, intents: set[str], ds: OasisDataset) -> None:
    info, st, s = res.infos[iid], res.stats[iid], res.series[iid]
    ch = res.changes[iid]
    mode = ch.attrs.get("mode", "abs")
    cl = {"pp": "pp", "pct": "%", "abs": ""}[mode]
    cov = f"{st['n']} observasi, {A.period_label(st['awal'])}–{A.period_label(st['akhir'])}"
    base = {"indicator_id": iid, "indikator": info["label_tampil"], "subsektor": info["sheet"],
            "satuan": A.unit_text(info), "cakupan": cov}
    out = []
    show_all = bool(intents & {"ringkasan", "bandingkan", "korelasi", "perhatian", "tren", "proyeksi"}) or not intents
    if "tertinggi" in intents:
        extra = f" (nilai sama muncul {st['n_maks']}×; ditampilkan yang pertama)" if st["n_maks"] > 1 else ""
        out.append({**base, "judul": "Nilai tertinggi", "nilai": st["maks"], "teks": A.fmt_val(st["maks"], info),
                    "periode": A.period_label(st["periode_maks"]),
                    "metode": f"Maksimum dari {st['n']} observasi bulanan pada jendela periode{extra}."})
    if "terendah" in intents:
        extra = f" (nilai sama muncul {st['n_min']}×; ditampilkan yang pertama)" if st["n_min"] > 1 else ""
        out.append({**base, "judul": "Nilai terendah", "nilai": st["min"], "teks": A.fmt_val(st["min"], info),
                    "periode": A.period_label(st["periode_min"]),
                    "metode": f"Minimum dari {st['n']} observasi bulanan{extra}."})
    if "rata_rata" in intents:
        out.append({**base, "judul": "Rata-rata", "nilai": st["rata_rata"], "teks": A.fmt_val(st["rata_rata"], info),
                    "periode": cov, "metode": "Rata-rata aritmetika nilai bulanan (bukan penjumlahan antarbulan)."})
    if "median" in intents:
        out.append({**base, "judul": "Median", "nilai": st["median"], "teks": A.fmt_val(st["median"], info),
                    "periode": cov, "metode": "Median nilai bulanan."})
    if "std" in intents:
        out.append({**base, "judul": "Simpangan baku", "nilai": st["std"], "teks": A.fmt_val(st["std"], info),
                    "periode": cov, "metode": "Simpangan baku sampel (ddof = 1) nilai bulanan."})
    if "kuartil" in intents:
        out.append({**base, "judul": "Kuartil (Q1 / Median / Q3)", "nilai": st["median"],
                    "teks": f"{A.fmt_num(st['q1'])} / {A.fmt_num(st['median'])} / {A.fmt_num(st['q3'])}"
                            f"{'' if A.unit_text(info) == 'satuan belum terkonfirmasi' else ' ' + A.unit_text(info)}",
                    "periode": cov, "metode": f"Kuartil metode interpolasi linier pandas; IQR {A.fmt_num(st['iqr'])}."})
    if "total" in intents:
        ok, why = A.sum_allowed(info, ds.flags.get(iid, []))
        if ok:
            tot = float(s.sum())
            out.append({**base, "judul": "Jumlah selama periode", "nilai": tot, "teks": A.fmt_val(tot, info),
                        "periode": cov, "metode": why})
        else:
            res.warnings.append(f"{res.label(iid)}: {why}")
            res.alternatives.append(f"Untuk {info['label_tampil']}: nilai akhir periode "
                                    f"({A.fmt_val(st['terakhir'], info)}, {A.period_label(st['akhir'])}) dan rata-rata "
                                    f"({A.fmt_val(st['rata_rata'], info)}) ditampilkan sebagai gantinya.")
            out.append({**base, "judul": "Nilai akhir periode (pengganti penjumlahan)", "nilai": st["terakhir"],
                        "teks": A.fmt_val(st["terakhir"], info), "periode": A.period_label(st["akhir"]),
                        "metode": "Posisi akhir jendela periode."})
    last_p = ch.index.max()
    if "mom" in intents or "yoy" in intents or show_all or "terakhir" in intents:
        out.append({**base, "judul": "Nilai terakhir", "nilai": st["terakhir"], "teks": A.fmt_val(st["terakhir"], info),
                    "periode": A.period_label(st["akhir"]), "metode": "Observasi terakhir pada jendela periode."})
    if "mom" in intents or show_all:
        v = ch["mom"].iloc[-1] if len(ch) else float("nan")
        if pd.notna(v):
            out.append({**base, "judul": "Perubahan MoM", "nilai": float(v), "teks": f"{'+' if v >= 0 else ''}{A.fmt_num(v)}{' pp' if cl == 'pp' else cl}",
                        "periode": f"{A.period_label(last_p - pd.offsets.MonthEnd(1))} → {A.period_label(last_p)}",
                        "metode": {"pp": "Selisih rasio dalam percentage points.", "pct": "Pertumbuhan % terhadap bulan sebelumnya.",
                                   "abs": "Selisih dalam satuan sumber (klasifikasi/satuan belum pasti)."}[mode]})
        elif "mom" in intents:
            res.warnings.append(f"{res.label(iid)}: MoM tidak dapat dihitung — bulan sebelumnya tidak tersedia.")
    if "yoy" in intents or show_all:
        v = ch["yoy"].iloc[-1] if len(ch) else float("nan")
        if pd.notna(v):
            out.append({**base, "judul": "Perubahan YoY", "nilai": float(v), "teks": f"{'+' if v >= 0 else ''}{A.fmt_num(v)}{' pp' if cl == 'pp' else cl}",
                        "periode": f"{A.period_label(last_p - pd.offsets.MonthEnd(12))} → {A.period_label(last_p)}",
                        "metode": {"pp": "Selisih terhadap bulan yang sama tahun lalu (pp).",
                                   "pct": "Pertumbuhan % terhadap bulan yang sama tahun lalu.",
                                   "abs": "Selisih terhadap bulan yang sama tahun lalu (satuan sumber)."}[mode]})
        elif "yoy" in intents:
            res.warnings.append(f"{res.label(iid)}: YoY tidak dapat dihitung — bulan yang sama tahun lalu tidak tersedia.")
    if "tren" in intents or show_all:
        tr = res.trends[iid]
        if tr.get("ok"):
            short = tr["arah"].split(" (")[0].replace("relatif datar", "datar")
            out.append({**base, "judul": "Tren linier", "nilai": tr["slope_bulan"], "teks": short,
                        "periode": tr["laju"] + (" · tidak signifikan" if "datar" in short else ""), "metode": tr["metode"] + f" R² = {A.fmt_num(tr['r2'])}."})
        elif "tren" in intents:
            res.warnings.append(f"{res.label(iid)}: {tr['alasan']}")
    res.answers += out


def _chart_options(res: AnalysisResult) -> None:
    n = len([i for i in res.ids if not res.series[i].empty])
    intents = set(res.query.intents)
    opts = []
    if n == 0:
        res.chart_options, res.default_chart = [], ""
        return
    if n == 1:
        opts = ["Garis", "Batang (nilai bulanan)", "Batang (perubahan YoY)", "Batang (perubahan MoM)"]
        default = "Garis"
        if "proyeksi" in intents:
            opts.insert(0, "Proyeksi")
            default = "Proyeksi"
    else:
        modes = res.compat["mode"] if res.compat else []
        if "sumbu_sama" in modes:
            opts.append("Garis (sumbu sama)")
        if "indeks_100" in modes:
            opts.append("Garis – indeks basis 100")
        if "perubahan" in modes or "indeks_100" in modes:
            opts.append("Garis – perubahan YoY")
        if n == 2:
            opts.append("Dua sumbu")
        opts.append("Berdampingan")
        if n == 2:
            opts.append("Scatter (eksploratif)")
        opts.append("Heatmap korelasi")
        if "sumbu_sama" in modes:
            opts.append("Batang (nilai terakhir)")
        default = opts[0]
        if "korelasi" in intents:
            default = "Scatter (eksploratif)" if n == 2 else "Heatmap korelasi"
        elif "sumbu_sama" not in modes and "indeks_100" in modes:
            default = "Garis – indeks basis 100"
        elif "sumbu_sama" not in modes and n == 2:
            default = "Dua sumbu"
        if "proyeksi" in intents:
            opts.insert(0, "Proyeksi")
            default = "Proyeksi"
        if "perhatian" in intents and n > 2:
            opts.insert(0, "Berdampingan")
            default = "Berdampingan"
        opts = list(dict.fromkeys(opts))
    res.chart_options, res.default_chart = opts, default


def provenance_table(ds: OasisDataset, res: AnalysisResult) -> pd.DataFrame:
    rows = []
    mp = ds.mapping.set_index("indicator_id") if not ds.mapping.empty else pd.DataFrame()
    for iid in res.ids:
        info = res.infos[iid]
        s = res.series[iid]
        m = mp.loc[iid] if iid in mp.index else {}
        rows.append({
            "sheet": info["sheet"], "field kamus": info["field_asli_kamus"],
            "kolom sheet": m.get("kolom_sheet", "-") if len(m) else "-",
            "metode pemetaan": m.get("metode", "-") if len(m) else "-",
            "label standar": info["label_tampil"], "satuan": A.unit_text(info),
            "status satuan": info["unit_status"], "jenis ukuran": info["measure_type"],
            "periode terpakai": f"{s.index.min():%Y-%m-%d} s.d. {s.index.max():%Y-%m-%d}" if not s.empty else "-",
            "jumlah observasi": len(s), "file": ds.provenance["nama_file"],
            "sha256": ds.provenance["sha256"][:12], "waktu unggah/penyusunan": ds.provenance["waktu_unggah"],
            "status data": ds.provenance["status_data"],
        })
    return pd.DataFrame(rows)
