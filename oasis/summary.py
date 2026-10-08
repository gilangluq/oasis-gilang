"""Penyusunan executive summary deterministik (templat) dari AnalysisResult.

Ringkasan hanya memakai angka yang sudah dihitung engine pada jendela periode
yang sama dengan grafik dan tabel.
"""
from __future__ import annotations

from . import analytics as A
from .engine import AnalysisResult


def build(res: AnalysisResult, ds_prov: dict) -> str:
    lines: list[str] = []
    ids = [i for i in res.ids if not res.series[i].empty]
    if not ids:
        return "Tidak ada data pada periode terpilih sehingga ringkasan tidak dapat disusun."
    lines.append(f"**Cakupan:** {res.window_note}")
    for iid in ids[:8]:
        info, st, tr = res.infos[iid], res.stats[iid], res.trends[iid]
        ch = res.changes[iid]
        mode = ch.attrs.get("mode", "abs")
        sfx = {"pp": " pp", "pct": "%", "abs": ""}[mode]
        parts = [f"posisi terakhir {A.fmt_val(st['terakhir'], info)} ({A.period_label(st['akhir'])})"]
        yoy = ch["yoy"].iloc[-1] if len(ch) else None
        if yoy is not None and yoy == yoy:
            parts.append(f"YoY {'+' if yoy >= 0 else ''}{A.fmt_num(yoy)}{sfx}")
        parts.append(f"rentang {A.fmt_num(st['min'])}–{A.fmt_num(st['maks'])} (puncak {A.period_label(st['periode_maks'])})")
        if tr.get("ok"):
            parts.append(f"tren {tr['arah'].split(' (')[0]}")
        lines.append(f"- **{res.label(iid)}**: " + "; ".join(parts) + f". ({st['n']} observasi)")
    if len(ids) > 8:
        lines.append(f"- … dan {len(ids) - 8} indikator lain (lihat tabel).")
    if res.compat:
        lines.append(f"**Komparabilitas:** {res.compat['label']} — {res.compat['alasan']}")
    for c in res.correlations[:3]:
        r = c.get("r_perubahan")
        if r is not None:
            lines.append(f"**Hubungan eksploratif** {res.label(c['a'])} vs {res.label(c['b'])}: korelasi perubahan "
                         f"({c['basis_perubahan']}) r = {A.fmt_num(r)} ({A.strength(r)}, n = {c['n_perubahan']}). "
                         f"Bukan bukti sebab-akibat.")
    for iid, fr in res.forecasts.items():
        if fr.ok:
            last = fr.prediksi.iloc[-1]
            lines.append(f"**Perkiraan {fr.horizon} bulan** {res.label(iid)} ({fr.metode_label}): "
                         f"{A.fmt_val(last['prediksi'], res.infos[iid])} pada {A.period_label(fr.prediksi.index[-1])}, "
                         f"interval ~80% {A.fmt_num(last['bawah'])}–{A.fmt_num(last['atas'])}. Ilustratif, bukan "
                         f"prediksi resmi OJK.")
        else:
            lines.append(f"**Proyeksi {res.label(iid)} tidak dibuat:** {fr.alasan}")
    hi = [a for a in res.attention if a["prioritas"] in ("Tinggi", "Sedang") and a["jenis"] != "kualitas"]
    if hi:
        lines.append(f"**Perlu perhatian ({len(hi)} butir prioritas tinggi/sedang):**")
        for a in hi[:4]:
            lines.append(f"  - [{a['prioritas']}] {a['indikator']} ({a['subsektor']}), {a['periode']}: {a['alasan']}")
    else:
        lines.append("**Perlu perhatian:** tidak ada pemicu prioritas tinggi/sedang pada jendela ini.")
    if res.warnings:
        lines.append(f"**Keterbatasan:** {len(res.warnings)} catatan (satuan/kualitas/komparabilitas) — lihat panel "
                     f"keterbatasan.")
    lines.append(f"_Sumber: {ds_prov['nama_file']} (sha256 {ds_prov['sha256'][:12]}), status data: "
                 f"{ds_prov['status_data']}, disusun {ds_prov['waktu_unggah']}. Threshold yang ditampilkan adalah "
                 f"threshold simulasi/konfigurasi pengguna, bukan batas regulasi._")
    return "\n\n".join(lines)
