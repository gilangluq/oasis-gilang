"""OASIS — OJK Agentic Self-Service Intelligence System (POC/MVP).

Jalankan:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from oasis import __version__
from oasis import analytics as A
from oasis import compat as C
from oasis import engine, llm, summary, viz
from oasis.attention import load_thresholds
from oasis.catalog import MEASURE_LABEL, UNIT_STATUS_LABEL
from oasis.data_prep import STATUS_DATA, TZ, load_workbook
from oasis.nlu import ParsedQuery, QuestionParser

ROOT = Path(__file__).resolve().parent
DEMO_PATH = ROOT / "data" / "demo" / "oasis_demo_simulasi.xlsx"
PUBLIC_DEMO = os.environ.get("OASIS_PUBLIC_DEMO", "0") == "1"

EXAMPLE_GROUPS = {
    "Perbandingan": [
        "Bandingkan tren Rasio NPL Bank Umum dan BPR 2023-2025",
        "Bandingkan total aset asuransi jiwa dan asuransi umum",
        "Total kredit dan NPL bank umum 24 bulan terakhir",
    ],
    "Statistik & perubahan": [
        "Berapa NPL tertinggi?",
        "Kuartil dan simpangan baku BOPO bank umum sejak 2023",
        "Rata-rata ROA perusahaan pembiayaan tahun 2024",
        "Perubahan YoY laba bank umum Juni 2026",
        "Jumlah kredit bank umum tahun 2024",
    ],
    "Lintas sektor": [
        "Bagaimana hubungan IHSG dan total investasi dana pensiun?",
    ],
    "Perlu perhatian": [
        "Apa yang perlu perhatian di perusahaan pembiayaan?",
        "Rasio klaim asuransi jiwa 12 bulan terakhir",
        "RBC asuransi",
    ],
    "Proyeksi": [
        "Proyeksi total aset BPR 12 bulan ke depan",
        "Proyeksi IHSG 6 bulan ke depan",
    ],
}
EXAMPLES = [q for qs in EXAMPLE_GROUPS.values() for q in qs]
EXAMPLE_LABEL = {q: f"{g} · {q}" for g, qs in EXAMPLE_GROUPS.items() for q in qs}
PRIVATE_DIR = ROOT / "data" / "private"
PUBLIC_WB = ROOT / "data" / "public" / "Dataset_dengan_Metadata_dan_Kamus_Data.xlsx"
FAVICON = ROOT / "assets" / "favicon.png"

LOGO_SVG = """<svg viewBox="0 0 64 64" width="{s}" height="{s}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<defs><linearGradient id="og" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0f2a5c"/>
<stop offset="1" stop-color="#1c5fae"/></linearGradient></defs>
<rect width="64" height="64" rx="14" fill="url(#og)"/>
<g fill="#fff" opacity=".93"><rect x="13" y="37.5" width="7.5" height="14" rx="1.8"/><rect x="24.5" y="29.5" width="7.5" height="22" rx="1.8"/>
<rect x="36" y="34" width="7.5" height="17.5" rx="1.8"/><rect x="47.5" y="20.5" width="7.5" height="31" rx="1.8"/></g>
<polyline points="16.8,32 28.2,24 39.8,28 51.2,14.5" fill="none" stroke="#5eead4" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
<g fill="#5eead4" stroke="#0f2a5c" stroke-width="1.2"><circle cx="16.8" cy="32" r="2.8"/><circle cx="28.2" cy="24" r="2.8"/>
<circle cx="39.8" cy="28" r="2.8"/><circle cx="51.2" cy="14.5" r="2.8"/></g></svg>"""


st.set_page_config(page_title="OASIS · Self-Service Intelligence", page_icon=str(FAVICON) if FAVICON.exists() else "📊",
                   layout="wide", initial_sidebar_state="auto")
st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1500px;}
.oasis-card {border:1px solid #e3e1da; border-radius:10px; padding:12px 14px; background:#fbfbf9; height:100%;}
.oasis-card .t {font-size:0.78rem; color:#6b6a66; text-transform:uppercase; letter-spacing:.03em;}
.oasis-card .v {font-size:1.35rem; font-weight:650; color:#1d1d1b; margin:2px 0;}
.oasis-card .m {font-size:0.8rem; color:#52514e;}
.att {border-left:4px solid #999; padding:8px 10px; margin-bottom:8px; background:#fbfbf9; border-radius:6px; font-size:0.86rem;}
.att.Tinggi {border-left-color:#c4471f;} .att.Sedang {border-left-color:#c98500;} .att.Rendah {border-left-color:#8a8984;}
.att .p {font-weight:700; font-size:0.75rem; text-transform:uppercase;}
.att.Tinggi .p {color:#c4471f;} .att.Sedang .p {color:#a26c00;} .att.Rendah .p {color:#6b6a66;}
.badge {display:inline-block; padding:1px 8px; border-radius:10px; font-size:0.75rem; background:#eef3fb; color:#1c4f8f; margin-right:4px;}
.badge.sim {background:#fff4dc; color:#8a5a00;}

/* ---------- Header ---------- */
.oasis-hero {position:relative; overflow:hidden; display:flex; align-items:center; justify-content:space-between;
  gap:24px; flex-wrap:wrap; padding:18px 24px; margin:0 0 6px 0; border-radius:16px; border:1px solid #e6e9ef;
  background:
    linear-gradient(90deg, rgba(28,95,174,.045) 1px, transparent 1px) 0 0/28px 28px,
    linear-gradient(0deg, rgba(28,95,174,.045) 1px, transparent 1px) 0 0/28px 28px,
    linear-gradient(120deg, #ffffff 0%, #f6f9fd 60%, #eef4fb 100%);}
.oasis-hero .brand {display:flex; align-items:center; gap:14px; z-index:1;}
.oasis-hero .wm {font-size:1.7rem; font-weight:700; letter-spacing:.14em; color:#0f2a5c; line-height:1;}
.oasis-hero .wm span {font-weight:400; color:#1c5fae; letter-spacing:.02em; font-size:1rem; margin-left:10px;}
.oasis-hero .tag {font-size:.82rem; color:#5b6472; margin-top:6px; letter-spacing:.01em;}
.oasis-hero .stats {display:flex; gap:22px; z-index:1; flex-wrap:wrap;}
.oasis-hero .stat {text-align:left;}
.oasis-hero .stat .n {font-size:1.15rem; font-weight:650; color:#0f2a5c; font-variant-numeric:tabular-nums;}
.oasis-hero .stat .l {font-size:.7rem; color:#6b7380; text-transform:uppercase; letter-spacing:.08em;}
.oasis-hero .spark {position:absolute; right:-10px; bottom:-6px; width:420px; height:90px; opacity:.55; z-index:0;
  pointer-events:none;}
.oasis-hero .pill {display:inline-block; font-size:.7rem; padding:2px 9px; border-radius:999px; margin-top:8px;
  background:#e8f0fb; color:#1c4f8f; letter-spacing:.03em;}
.oasis-hero .pill.sim {background:#fff4dc; color:#8a5a00;}
@media (max-width: 760px) {.oasis-hero .spark {display:none;} .oasis-hero .wm span {display:block; margin:6px 0 0 0;}}

/* ---------- Ask box (primary interaction) ---------- */
.st-key-askbox {background:linear-gradient(135deg,#0f2a5c 0%,#1c5fae 100%); border-radius:18px;
  padding:22px 24px 14px 24px; box-shadow:0 10px 28px -14px rgba(15,42,92,.55);}
.st-key-askbox .ask-title {color:#fff; font-size:1.25rem; font-weight:650; margin:0 0 2px 0;}
.st-key-askbox .ask-sub {color:#c9d8ee; font-size:.86rem; margin:0 0 10px 0;}
.st-key-askbox [data-testid="stTextInputRootElement"] {border:2px solid #5eead4 !important;
  border-radius:12px !important; background:#fff !important; min-height:3.4rem;}
.st-key-askbox [data-testid="stTextInput"] input {font-size:1.12rem !important; padding:0.85rem 1rem !important;
  color:#0f1b2d !important;}
.st-key-askbox [data-testid="stTextInputRootElement"]:focus-within {box-shadow:0 0 0 4px rgba(94,234,212,.40);}
.st-key-askbox [data-testid="stBaseButton-primary"] {min-height:3.4rem; border-radius:12px; font-size:1.02rem;
  font-weight:650; background:#5eead4 !important; color:#0f2a5c !important; border:none !important;}
.st-key-askbox [data-testid="stBaseButton-primary"]:hover {background:#99f6e4 !important;}
.st-key-askbox [data-testid="stWidgetLabel"] p {color:#dbe6f5 !important; font-size:.82rem;}
.st-key-askbox [data-testid="stSelectbox"] [data-baseweb="select"] > div {border-radius:10px; background:rgba(255,255,255,.96);}
.section-title {font-size:1.05rem; font-weight:650; color:#0f2a5c; margin:18px 0 4px 0; padding-bottom:4px;
  border-bottom:1px solid #e6e9ef;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# State & data
# ---------------------------------------------------------------------------
def _init_state():
    ss = st.session_state
    ss.setdefault("unit_conf", {})
    ss.setdefault("th_override", {})
    ss.setdefault("audit", [])
    ss.setdefault("q", EXAMPLES[0])
    ss.setdefault("sel", None)  # {"q": pertanyaan, "ids": [...]}


def log(action: str, detail: str):
    st.session_state.audit.append({"waktu": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"), "aksi": action,
                                   "detail": detail})


@st.cache_data(show_spinner="Membaca workbook, membangun katalog, dan memvalidasi data…")
def _load(content: bytes, name: str, status: str, _hash: str):
    return load_workbook(content, name, status)


_init_state()
ss = st.session_state

with st.sidebar:
    st.markdown(f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:2px">{LOGO_SVG.format(s=30)}'
                f'<span style="font-weight:700;letter-spacing:.14em;color:#0f2a5c;font-size:1.05rem">OASIS</span></div>',
                unsafe_allow_html=True)
    st.caption("OJK Agentic Self-Service Intelligence System — POC/MVP v" + __version__)
    LOCAL_WB = os.environ.get("OASIS_LOCAL_WORKBOOK")
    if not LOCAL_WB and not PUBLIC_DEMO and PRIVATE_DIR.exists():
        cands = sorted(PRIVATE_DIR.glob("*.xlsx"))
        LOCAL_WB = str(cands[0]) if cands else None
    sources = []
    if PUBLIC_WB.exists():
        sources.append("Data statistik publik (sumber utama)")
    if LOCAL_WB and Path(LOCAL_WB).exists() and not PUBLIC_DEMO:
        sources.append(f"Workbook lokal ({Path(LOCAL_WB).name})")
    sources += ["Demo simulasi (aman dipublikasikan)", "Unggah workbook"]
    src = st.radio("Sumber data", sources, index=0)
    content, fname = None, None
    if src.startswith("Data statistik publik"):
        content, fname = PUBLIC_WB.read_bytes(), PUBLIC_WB.name
        status = "Publik"
        st.markdown('<span class="badge">DATA PUBLIK</span>', unsafe_allow_html=True)
    elif src.startswith("Workbook lokal"):
        content, fname = Path(LOCAL_WB).read_bytes(), Path(LOCAL_WB).name
        status = st.selectbox("Status data", STATUS_DATA, index=STATUS_DATA.index("Belum dinyatakan"))
    elif src.startswith("Demo"):
        if not DEMO_PATH.exists():
            st.error("Workbook demo belum ada. Jalankan: python scripts/generate_demo_workbook.py")
            st.stop()
        content, fname = DEMO_PATH.read_bytes(), DEMO_PATH.name
        status = "Simulasi"
        st.markdown('<span class="badge sim">DATA SIMULASI</span>', unsafe_allow_html=True)
    else:
        if PUBLIC_DEMO:
            st.warning("Mode demo publik: JANGAN mengunggah data internal/rahasia OJK. Gunakan data simulasi/publik.")
        up = st.file_uploader("Workbook (.xlsx) dengan sheet subsektor + Kamus_Data", type=["xlsx"])
        status = st.selectbox("Status data yang diunggah", STATUS_DATA, index=STATUS_DATA.index("Belum dinyatakan"),
                              help="Dicatat pada asal data setiap hasil.")
        if up is None:
            st.info("Unggah workbook untuk mulai, atau pilih data demo.")
            st.stop()
        content, fname = up.getvalue(), up.name
        if PUBLIC_DEMO and status.startswith("Asli"):
            st.error("Data berstatus internal tidak boleh dipakai pada demo publik.")
            st.stop()

    try:
        base_ds = _load(content, fname, status, hashlib.sha256(content).hexdigest())
    except Exception as e:  # noqa: BLE001
        st.error(f"Workbook tidak dapat diproses: {e}")
        st.stop()
    ds = base_ds.with_unit_confirmations(ss.unit_conf)

    dmin, dmax = ds.period_bounds()
    months = list(pd.date_range(dmin, dmax, freq="ME"))
    rng = st.select_slider("Filter periode (berlaku untuk semua tab)", options=months, value=(months[0], months[-1]),
                           format_func=A.period_label)
    gstart, gend = rng

    use_llm = False
    if llm.available():
        use_llm = st.toggle("Tafsir pertanyaan dengan LLM (opsional)", value=False,
                            help="Hanya teks pertanyaan + daftar id katalog yang dikirim; perhitungan tetap deterministik.")
    else:
        st.caption("LLM: tidak aktif (tanpa API key) — parser aturan dipakai.")

    with st.expander("⚖️ Konfirmasi satuan", expanded=False):
        unconf = ds.base_catalog[ds.base_catalog["unit_status"] == "belum_terkonfirmasi"]
        st.caption(f"{len(unconf)} indikator berstatus 'satuan belum terkonfirmasi'. Konfirmasi hanya bila ada dasar "
                   f"yang dapat diaudit; tercatat di log tata kelola.")
        opt = st.selectbox("Indikator", unconf["indicator_id"].tolist(), key="uc_ind",
                           format_func=lambda i: f"{ds.info(i)['label_tampil']} — {ds.info(i)['sheet']}")
        cu1, cu2 = st.columns(2)
        unit_in = cu1.text_input("Satuan", placeholder="mis. Rp miliar")
        who = cu2.text_input("Dikonfirmasi oleh", placeholder="nama/unit")
        basis = st.text_input("Dasar konfirmasi", placeholder="mis. dokumen definisi data v1.2")
        if st.button("Simpan konfirmasi", width="stretch", disabled=not (opt and unit_in and who and basis)):
            ss.unit_conf[opt] = {"unit": unit_in.strip(), "oleh": who.strip(), "dasar": basis.strip(),
                                 "waktu": datetime.now(TZ).strftime("%Y-%m-%d %H:%M WIB")}
            log("konfirmasi_satuan", f"{opt} → {unit_in} oleh {who} (dasar: {basis})")
            st.rerun()
        if ss.unit_conf:
            st.dataframe(pd.DataFrame([{"indikator": k, **v} for k, v in ss.unit_conf.items()]), hide_index=True)
            if st.button("Hapus semua konfirmasi", width="stretch"):
                log("hapus_konfirmasi_satuan", ", ".join(ss.unit_conf))
                ss.unit_conf = {}
                st.rerun()

    with st.expander("📏 Threshold (konfigurasi pengguna)", expanded=False):
        st.caption("Default: threshold SIMULASI dari config/thresholds.json — bukan batas regulasi OJK.")
        th_id = st.selectbox("Indikator ", ds.catalog["indicator_id"].tolist(), key="th_id",
                             format_func=lambda i: f"{ds.info(i)['label_tampil']} — {ds.info(i)['sheet']}")
        t1, t2 = st.columns(2)
        th_dir = t1.selectbox("Arah", ["atas", "bawah"], help="atas = perhatian bila nilai > batas")
        th_val = t2.number_input("Nilai batas", value=0.0, format="%.4f")
        th_src = st.text_input("Sumber/dasar", value="konfigurasi pengguna")
        if st.button("Terapkan threshold", width="stretch"):
            ss.th_override[th_id] = [{"nilai": float(th_val), "arah": th_dir,
                                      "label": f"{ds.info(th_id)['label_tampil']} {th_val:g} (konfigurasi pengguna)",
                                      "sumber": th_src or "konfigurasi pengguna"}]
            log("threshold", f"{th_id} {th_dir} {th_val} ({th_src})")
            st.rerun()
        if ss.th_override and st.button("Kembalikan ke default", width="stretch"):
            ss.th_override = {}
            log("threshold_reset", "-")
            st.rerun()

parser = QuestionParser(ds.catalog)


# ---------------------------------------------------------------------------
# Komponen render
# ---------------------------------------------------------------------------
def card(title: str, value: str, meta: str, help_: str = ""):
    st.markdown(f'<div class="oasis-card" title="{help_}"><div class="t">{title}</div><div class="v">{value}</div>'
                f'<div class="m">{meta}</div></div>', unsafe_allow_html=True)


def render_answers(res: engine.AnalysisResult):
    ans = res.answers
    if not ans:
        return
    if len(res.ids) <= 3 and len(ans) <= 18:
        for iid in res.ids:
            group = [a for a in ans if a["indicator_id"] == iid]
            if not group:
                continue
            st.markdown(f"**{res.label(iid)}**")
            cols = st.columns(max(4, min(6, len(group))))
            for k, a in enumerate(group):
                with cols[k % len(cols)]:
                    card(a["judul"], a["teks"], f"{a['periode']}<br><i>{a['cakupan']}</i>", a["metode"])
        with st.expander("Metode perhitungan tiap angka"):
            st.dataframe(pd.DataFrame(ans)[["judul", "indikator", "subsektor", "teks", "satuan", "periode", "cakupan",
                                            "metode"]], hide_index=True)
    else:
        st.dataframe(pd.DataFrame(ans)[["subsektor", "indikator", "judul", "teks", "satuan", "periode", "cakupan",
                                        "metode"]].rename(columns={"teks": "nilai"}), hide_index=True, height=320)


def render_attention(res: engine.AnalysisResult, max_items: int = 8):
    st.markdown("#### 🚩 Perlu Perhatian")
    items = res.attention
    if not items:
        st.success("Tidak ada pemicu pada jendela periode ini.")
        return
    main = [a for a in items if a["jenis"] != "kualitas"]
    qual = [a for a in items if a["jenis"] == "kualitas"]
    if not main:
        st.success("Tidak ada pemicu threshold/anomali/tren memburuk.")
    for a in main[:max_items]:
        st.markdown(f'<div class="att {a["prioritas"]}"><span class="p">{a["prioritas"]}</span> · {a["jenis"]}<br>'
                    f'<b>{a["indikator"]}</b> ({a["subsektor"]}) — {a["periode"]}<br>{a["alasan"]}<br>'
                    f'<span style="color:#52514e">Angka: {a["angka"]}</span><br>'
                    f'<span style="color:#1c4f8f">Saran pemeriksaan: {a["tindakan"]}</span></div>',
                    unsafe_allow_html=True)
    if len(main) > max_items:
        st.caption(f"+{len(main) - max_items} butir lain di tabel lengkap.")
    if qual:
        st.caption(f"{len(qual)} catatan kualitas data (prioritas rendah) di tabel lengkap.")
    with st.expander("Tabel lengkap Perlu Perhatian"):
        st.dataframe(pd.DataFrame(items)[["prioritas", "jenis", "subsektor", "indikator", "periode", "alasan", "angka",
                                          "tindakan"]], hide_index=True)


def chart_for(res: engine.AnalysisResult, kind: str, fc_id: str | None = None):
    """Kembalikan (figure, dataframe grid) — grid = data persis yang diplot."""
    ids = [i for i in res.ids if not res.series[i].empty]
    base_tbl = res.table()
    if kind in ("Garis", "Garis (sumbu sama)"):
        return viz.line_chart(res), base_tbl
    if kind == "Garis – indeks basis 100":
        idx, base = A.index_base100(res.aligned["irisan"])
        g = idx.rename(columns=lambda c: res.label(c)).reset_index()
        g["periode"] = g["periode"].dt.strftime("%Y-%m-%d")
        g.insert(1, "basis", A.period_label(base))
        return viz.line_chart(res, "indeks"), g
    if kind == "Garis – perubahan YoY":
        return viz.line_chart(res, "yoy"), base_tbl[["periode", "subsektor", "indikator", "YoY", "satuan_perubahan"]]
    if kind == "Dua sumbu":
        return viz.dual_axis(res), base_tbl
    if kind == "Berdampingan":
        return viz.small_multiples(res), base_tbl
    if kind == "Scatter (eksploratif)":
        a, b = ids[:2]
        df = pd.concat([res.series[a].rename(res.label(a)), res.series[b].rename(res.label(b))], axis=1).dropna()
        df.index = df.index.strftime("%Y-%m-%d")
        return viz.scatter(res), df.reset_index(names="periode")
    if kind == "Heatmap korelasi":
        fig, corr = viz.corr_heatmap(res)
        return fig, corr.reset_index(names="indikator")
    if kind == "Batang (nilai terakhir)":
        return viz.bar_last(res), base_tbl
    if kind.startswith("Batang"):
        k = "yoy" if "YoY" in kind else "mom" if "MoM" in kind else "nilai"
        return viz.bar_single(res, k), base_tbl
    if kind == "Proyeksi":
        iid = fc_id or next(iter(res.forecasts))
        fr = res.forecasts[iid]
        g = base_tbl[base_tbl["indikator"] == res.infos[iid]["label_tampil"]][["periode", "nilai", "satuan"]].copy()
        g["jenis"] = "historis"
        if fr.ok:
            p = fr.prediksi.reset_index(names="periode")
            p["periode"] = p["periode"].dt.strftime("%Y-%m-%d")
            p = p.rename(columns={"prediksi": "nilai", "bawah": "batas_bawah_80", "atas": "batas_atas_80"})
            p["satuan"] = A.unit_text(res.infos[iid])
            p["jenis"] = f"perkiraan ({fr.metode_label})"
            g = pd.concat([g, p], ignore_index=True)
        return viz.forecast_chart(res, iid), g
    return viz.line_chart(res), base_tbl


def render_chart_block(res: engine.AnalysisResult, key: str):
    if not res.chart_options:
        return
    left, right = st.columns([2.6, 1.2], gap="large")
    with left:
        c1, c2 = st.columns([2, 1])
        kind = c1.selectbox("Jenis grafik (hanya pilihan yang valid untuk data ini)", res.chart_options,
                            index=res.chart_options.index(res.default_chart),
                            key=f"ct_{key}_{abs(hash((tuple(res.ids), tuple(res.chart_options)))) % 10**8}")
        fc_id = None
        if kind == "Proyeksi" and len(res.forecasts) > 1:
            fc_id = c2.selectbox("Seri proyeksi", list(res.forecasts), format_func=res.label, key=f"fc_{key}")
        if res.compat:
            c2.markdown(f'<span class="badge">Kompatibilitas: {res.compat["label"]}</span>', unsafe_allow_html=True)
        fig, grid = chart_for(res, kind, fc_id)
        tg, td = st.tabs(["📊 Grafik", "🧾 Data (grid)"])
        with tg:
            ev = st.plotly_chart(fig, key=f"pc_{key}_{kind}", on_select="rerun", selection_mode="points",
                                 config={"displaylogo": False})
            if kind == "Scatter (eksploratif)":
                st.caption("Eksploratif — pola bersama tidak menunjukkan sebab-akibat atau transmisi risiko. Warna titik "
                           "= waktu; garis OLS hanya deskriptif.")
            if kind == "Dua sumbu":
                st.caption("⚠️ Dua sumbu dipakai karena satuan berbeda. Tinggi garis antarsumbu tidak dapat dibandingkan "
                           "langsung; untuk pola pertumbuhan gunakan 'Berdampingan' atau indeks bila valid.")
            _anomaly_click(res, fig, ev)
        with td:
            st.dataframe(grid, hide_index=True, height=360)
            st.download_button("Unduh grid (CSV)", grid.to_csv(index=False).encode("utf-8"),
                               file_name=f"oasis_{key}_{kind[:12]}.csv".replace(" ", "_"), key=f"dl_{key}_{kind}")
        if any(not v.empty for v in res.anomalies.values()):
            with st.expander("🔴 Daftar anomali & metode deteksi"):
                rows = []
                for iid, an in res.anomalies.items():
                    for _, a in an.iterrows():
                        rows.append({"indikator": res.label(iid), "periode": A.period_label(a["periode"]),
                                     "nilai": A.fmt_val(a["nilai"], res.infos[iid]), "skor_z": round(a["skor_z"], 2),
                                     "alasan": a["alasan"]})
                st.dataframe(pd.DataFrame(rows), hide_index=True)
                for iid, m in res.anomaly_method.items():
                    st.caption(f"**{res.label(iid)}** — {m}")
    with right:
        render_attention(res)


def _anomaly_click(res, fig, ev):
    pts = []
    try:
        pts = ev.selection.points if ev and ev.selection else []
    except AttributeError:
        pts = (ev or {}).get("selection", {}).get("points", [])
    shown = False
    for p in pts:
        cn = p.get("curve_number")
        pi = p.get("point_index", p.get("point_number"))
        if cn is None or pi is None or cn >= len(fig.data):
            continue
        tr = fig.data[cn]
        if tr.name != "Anomali (klik)":
            continue
        cd = tr.customdata[pi]
        iid = cd[5]
        method = res.anomaly_method.get(iid, "")
        st.error(f"**Anomali terpilih — {cd[3]} ({cd[4]}), {cd[0]}: {cd[1]}**\n\n{cd[2]}\n\n_Metode:_ {method}")
        shown = True
    if not shown and any(t.name == "Anomali (klik)" for t in fig.data):
        st.caption("🔴 Klik titik merah pada grafik untuk melihat alasan dan metode deteksi anomali.")


def render_forecasts(res: engine.AnalysisResult):
    for iid, fr in res.forecasts.items():
        st.markdown(f"##### 📈 Proyeksi — {res.label(iid)}")
        if not fr.ok:
            st.warning(f"Proyeksi tidak dibuat: {fr.alasan}")
            continue
        c = st.columns(4)
        last = fr.prediksi.iloc[-1]
        with c[0]:
            card("Perkiraan akhir horizon", A.fmt_val(last["prediksi"], res.infos[iid]), A.period_label(fr.prediksi.index[-1]))
        with c[1]:
            card("Interval perkiraan ~80%", f"{A.fmt_num(last['bawah'])} – {A.fmt_num(last['atas'])}", "kuantil galat backtest")
        with c[2]:
            card("Metode terpilih", fr.metode_label, f"horizon {fr.horizon} bulan")
        with c[3]:
            card("Data latih", f"{fr.n_latih} observasi", f"{A.period_label(fr.latih_awal)}–{A.period_label(fr.latih_akhir)}")
        if fr.evaluasi is not None:
            st.markdown("Evaluasi backtest rolling-origin (semakin kecil semakin baik):")
            st.dataframe(fr.evaluasi.drop(columns=["metode"]).round(3), hide_index=True)
        for n in fr.catatan:
            st.caption("• " + n)


def render_result(res: engine.AnalysisResult, key: str):
    st.caption("🗓️ " + res.window_note)
    # 1) Visualisasi menjadi fokus utama tepat setelah panel pemahaman pertanyaan
    st.markdown('<div class="section-title">📊 Visualisasi & Perlu Perhatian</div>', unsafe_allow_html=True)
    render_chart_block(res, key)
    for n in res.query.notes:
        st.caption("ℹ️ " + n)
    if res.forecasts:
        render_forecasts(res)
    # 2) Angka kunci
    if res.answers:
        st.markdown('<div class="section-title">🔢 Angka kunci</div>', unsafe_allow_html=True)
        render_answers(res)
    # 3) Rincian metodologis
    if res.warnings or res.alternatives:
        with st.expander(f"⚠️ Keterbatasan, satuan & komparabilitas ({len(res.warnings)})", expanded=bool(res.alternatives)):
            for w in dict.fromkeys(res.warnings):
                st.markdown(f"- {w}")
            if res.alternatives:
                st.markdown("**Alternatif analisis yang valid:**")
                for a in dict.fromkeys(res.alternatives):
                    st.markdown(f"- {a}")
    if res.compat:
        with st.expander("🔀 Kompatibilitas & penyelarasan periode"):
            st.markdown(f"**{res.compat['label']}** — {res.compat['alasan']}")
            if res.aligned:
                ri = res.aligned["rentang_irisan"]
                st.markdown(f"Periode beririsan: **{res.aligned['n_irisan']}** bulan "
                            f"({A.period_label(ri[0])}–{A.period_label(ri[1])}).")
                miss = {res.label(k): ", ".join(A.period_label(p) for p in v) for k, v in res.aligned["hilang"].items() if v}
                st.markdown("Periode hilang: " + ("; ".join(f"{k}: {v}" for k, v in miss.items()) if miss else "tidak ada."))
            st.dataframe(pd.DataFrame([{"A": res.label(p["a"]), "B": res.label(p["b"]), "level": p["label"],
                                        "alasan": p["alasan"]} for p in res.pairs]), hide_index=True)
    if res.correlations:
        with st.expander("🔗 Hubungan eksploratif (bukan sebab-akibat)", expanded="korelasi" in res.query.intents):
            rows = []
            for c in res.correlations:
                rows.append({"A": res.label(c["a"]), "B": res.label(c["b"]),
                             "r level": round(c.get("r_level", float("nan")), 3), "n level": c["n_level"],
                             "r perubahan": round(c.get("r_perubahan", float("nan")), 3),
                             "basis perubahan": c["basis_perubahan"], "n perubahan": c["n_perubahan"],
                             "kekuatan (perubahan)": A.strength(c.get("r_perubahan"))})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption(res.correlations[0]["catatan"])
    # 4) Ringkasan eksekutif & asal data
    st.markdown('<div class="section-title">🧭 Executive summary</div>', unsafe_allow_html=True)
    summ = summary.build(res, ds.provenance)
    with st.container(border=True):
        st.markdown(summ)
    with st.expander("🧾 Asal data (provenance) untuk hasil ini"):
        st.dataframe(res.provenance, hide_index=True)
    c1, c2 = st.columns(2)
    c1.download_button("Unduh tabel data pendukung (CSV)", res.table().to_csv(index=False).encode("utf-8"),
                       file_name=f"oasis_{key}_data.csv", key=f"dlt_{key}")
    c2.download_button("Unduh ringkasan (Markdown)", (summ + "\n\n## Asal data\n\n" + _md_table(res.provenance)).encode("utf-8"),
                       file_name=f"oasis_{key}_ringkasan.md", key=f"dls_{key}")


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(out)


def manual_query(intents: list[str], horizon: int | None = None) -> ParsedQuery:
    pq = ParsedQuery(question="(pilihan manual)", normalized="", intents=intents, horizon=horizon)
    pq.source = "pilihan manual"
    return pq


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
n_crit = sum(1 for i in ds.issues if i.tingkat == "Kritis")
n_warn = sum(1 for i in ds.issues if i.tingkat == "Peringatan")
_spark = ("<svg class='spark' viewBox='0 0 420 90' preserveAspectRatio='none' aria-hidden='true'>"
          "<defs><linearGradient id='sf' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#1c5fae' "
          "stop-opacity='.16'/><stop offset='1' stop-color='#1c5fae' stop-opacity='0'/></linearGradient></defs>"
          "<path d='M0,70 L40,62 L80,66 L120,50 L160,55 L200,38 L240,44 L280,30 L320,34 L360,18 L420,22 L420,90 L0,90Z' "
          "fill='url(#sf)'/><polyline points='0,70 40,62 80,66 120,50 160,55 200,38 240,44 280,30 320,34 360,18 420,22' "
          "fill='none' stroke='#1c5fae' stroke-opacity='.35' stroke-width='1.6'/></svg>")
_pill_cls = "pill sim" if ds.provenance["status_data"] == "Simulasi" else "pill"
st.markdown(
    f"""<div class="oasis-hero">{_spark}
  <div class="brand">{LOGO_SVG.format(s=46)}
    <div><div class="wm">OASIS<span>Self-Service Intelligence</span></div>
    <div class="tag">Analitik pengawasan lintas sektor jasa keuangan · bahasa Indonesia · perhitungan deterministik</div>
    <span class="{_pill_cls}">Sumber: {ds.provenance['nama_file']} · status: {ds.provenance['status_data']}</span></div>
  </div>
  <div class="stats">
    <div class="stat"><div class="n">{len(ds.catalog)}</div><div class="l">indikator</div></div>
    <div class="stat"><div class="n">{ds.catalog['sheet'].nunique()}</div><div class="l">subsektor</div></div>
    <div class="stat"><div class="n">{A.period_label(dmin)}–{A.period_label(dmax)}</div><div class="l">cakupan periode</div></div>
    <div class="stat"><div class="n">{n_crit} · {n_warn}</div><div class="l">kritis · peringatan</div></div>
  </div>
</div>""",
    unsafe_allow_html=True,
)

tabs = st.tabs(["💬 Tanya OASIS", "📚 Katalog Data", "🧪 Kualitas Data", "🔀 Kompatibilitas", "🕸️ Lintas Sektor",
                "📈 Proyeksi", "ℹ️ Tata Kelola"])

# ---------------------------------------------------------------------------
# Tab 1 — Tanya
# ---------------------------------------------------------------------------
with tabs[0]:
    def _use_example():
        ex = ss.get("ex_pick")
        if ex:
            ss.q = ex
            ss.sel = None

    with st.container(key="askbox"):
        st.markdown('<div class="ask-title">Tanyakan pada OASIS</div>'
                    '<div class="ask-sub">Tulis pertanyaan analitik dalam bahasa Indonesia — sebut indikator, subsektor, '
                    'dan periode bila perlu. Tekan Enter atau klik Analisis.</div>', unsafe_allow_html=True)
        qc1, qc2 = st.columns([6, 1], vertical_alignment="bottom")
        q = qc1.text_input("Pertanyaan", key="q", label_visibility="collapsed",
                           placeholder="mis. Bandingkan tren Rasio NPL Bank Umum dan BPR 2023-2025")
        qc2.button("Analisis", type="primary", width="stretch", icon=":material/insights:")
        st.selectbox("💡 Contoh pertanyaan — pilih untuk mengisi kolom di atas", EXAMPLES, index=None,
                     key="ex_pick", on_change=_use_example, format_func=lambda x: EXAMPLE_LABEL.get(x, x),
                     placeholder="Pilih contoh pertanyaan…")
    if ss.sel and ss.sel.get("q") != q:
        ss.sel = None

    pq = llm.interpret(q, ds.catalog, parser, dmax) if use_llm else parser.parse(q, dmax)
    with st.expander("🧠 Bagaimana OASIS memahami pertanyaan ini"):
        for t in pq.trace:
            st.markdown(f"- {t}")
        st.json(pq.to_dict(), expanded=False)

    ids = list(pq.resolved)
    if ss.sel and ss.sel.get("q") == q:
        ids = list(dict.fromkeys(ids + ss.sel["ids"]))
    pending = pq.ambiguities and not (ss.sel and ss.sel.get("q") == q)
    if pending:
        for n in pq.notes:
            st.info(n)
        st.warning("Pertanyaan ambigu — OASIS tidak memilih diam-diam. Silakan pilih indikator/subsektor yang dimaksud:")
        for ai, amb in enumerate(pq.ambiguities):
            st.markdown(f"**{amb['pesan']}**")
            opts = amb["pilihan"]
            bc = st.columns(min(4, len(opts)) or 1)
            for oi, o in enumerate(opts[:24]):
                if bc[oi % len(bc)].button(parser.label(o), key=f"amb{ai}_{oi}", width="stretch"):
                    ss.sel = {"q": q, "ids": [o]}
                    log("pilih_ambiguitas", f"{q} → {o}")
                    st.rerun()
            if len(opts) > 1:
                if st.button(f"Gunakan semua ({len(opts)}) untuk dibandingkan", key=f"amb_all{ai}"):
                    ss.sel = {"q": q, "ids": opts}
                    log("pilih_ambiguitas", f"{q} → semua {len(opts)}")
                    st.rerun()
    if ids and not pending:
        intents = pq.intents
        if len(ids) > 1 and not set(intents) & {"bandingkan", "korelasi", "perhatian", "ringkasan", "proyeksi"}:
            pq.intents = intents + ["bandingkan"]
        res = engine.run(ds, pq, ids, gstart, gend, ss.th_override)
        render_result(res, "tanya")
    elif not ids and not pending:
        for n in pq.notes:
            st.info(n)
        st.markdown("**Indikator yang dapat ditanyakan** (dari Kamus_Data):")
        st.dataframe(ds.catalog.groupby("concept_name")["sheet"].apply(lambda s: ", ".join(sorted(set(s))))
                     .reset_index().rename(columns={"concept_name": "indikator", "sheet": "tersedia di"}),
                     hide_index=True, height=300)

# ---------------------------------------------------------------------------
# Tab 2 — Katalog
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("### Katalog data (sumber kebenaran: Kamus_Data)")
    st.caption("Setiap indikator berasal dari satu baris Kamus_Data. OASIS tidak menambah field. Kolom 'kolom sheet' "
               "menunjukkan header sumber yang dipetakan beserta metodenya.")
    k1, k2, k3 = st.columns(3)
    f_sheet = k1.multiselect("Subsektor", sorted(ds.catalog["sheet"].unique()))
    f_unit = k2.multiselect("Status satuan", list(UNIT_STATUS_LABEL.values()))
    f_type = k3.multiselect("Jenis ukuran", list(MEASURE_LABEL.values()))
    cat = ds.catalog.merge(ds.mapping[["indicator_id", "kolom_sheet", "metode", "keyakinan"]], on="indicator_id", how="left")
    cat = cat.assign(status_satuan=cat["unit_status"].map(UNIT_STATUS_LABEL),
                     jenis=cat["measure_type"].map(MEASURE_LABEL),
                     satuan=cat.apply(lambda r: A.unit_text(r.to_dict()), axis=1))
    if f_sheet:
        cat = cat[cat["sheet"].isin(f_sheet)]
    if f_unit:
        cat = cat[cat["status_satuan"].isin(f_unit)]
    if f_type:
        cat = cat[cat["jenis"].isin(f_type)]
    obs = ds.long.dropna().groupby("indicator_id")["periode"].agg(["min", "max", "count"])
    cat = cat.join(obs, on="indicator_id")
    st.dataframe(cat[["sheet", "field_asli_kamus", "label_tampil", "kolom_sheet", "metode", "deskripsi", "tipe_data",
                      "unit_raw", "satuan", "status_satuan", "jenis", "measure_note", "frekuensi", "mulai_kamus",
                      "akhir_kamus", "min", "max", "count", "concept_name"]]
                 .rename(columns={"field_asli_kamus": "nama field asli (kamus)", "label_tampil": "label standar",
                                  "kolom_sheet": "kolom sheet", "metode": "metode pemetaan", "unit_raw": "unit (kamus)",
                                  "measure_note": "dasar klasifikasi", "mulai_kamus": "mulai (kamus)",
                                  "akhir_kamus": "akhir (kamus)", "min": "data mulai", "max": "data akhir",
                                  "count": "observasi", "concept_name": "konsep parser"}),
                 hide_index=True, height=520)
    st.download_button("Unduh katalog (CSV)", cat.to_csv(index=False).encode("utf-8"), file_name="oasis_katalog.csv")
    if ds.metadata_sheet is not None:
        with st.expander("Metadata_Sheet (ringkasan per sheet dari sumber)"):
            st.dataframe(ds.metadata_sheet, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 3 — Kualitas
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("### Laporan kualitas data")
    issues = ds.issues_df()
    m = st.columns(5)
    m[0].metric("Sheet ditemukan", f"{int(ds.sheet_summary['ditemukan'].sum())}/{len(ds.sheet_summary)}")
    m[1].metric("Temuan kritis", int((issues["tingkat"] == "Kritis").sum()))
    m[2].metric("Peringatan", int((issues["tingkat"] == "Peringatan").sum()))
    m[3].metric("Info", int((issues["tingkat"] == "Info").sum()))
    mapped = ds.mapping.dropna(subset=["indicator_id"])
    m[4].metric("Field cocok persis", f"{int((mapped['metode'] == 'cocok persis').sum())}/{len(mapped)}")
    st.markdown("**Ringkasan per sheet**")
    st.dataframe(ds.sheet_summary, hide_index=True)
    st.markdown("**Temuan** (urut: Kritis → Peringatan → Info)")
    lv = st.multiselect("Tingkat", ["Kritis", "Peringatan", "Info"], default=["Kritis", "Peringatan", "Info"])
    st.dataframe(issues[issues["tingkat"].isin(lv)], hide_index=True, height=380)
    st.markdown("**Uji aturan eksplisit** (config/relations.json) — dasar graph/network dan konsistensi skala")
    st.dataframe(ds.rule_results[["id", "sheet", "jenis", "target", "referensi", "status", "n", "lolos_pct", "detail",
                                  "dasar"]], hide_index=True)
    with st.expander("Tabel pemetaan field kamus ↔ header sheet"):
        st.dataframe(ds.mapping, hide_index=True)
    st.download_button("Unduh laporan kualitas (CSV)", issues.to_csv(index=False).encode("utf-8"),
                       file_name="oasis_kualitas.csv")

# ---------------------------------------------------------------------------
# Tab 4 — Kompatibilitas
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown("### Matriks kompatibilitas indikator")
    st.markdown(
        "- **Absolut**: konsep, jenis ukuran, dan satuan sama serta terkonfirmasi → boleh satu sumbu.\n"
        "- **Perubahan (pp)**: sama-sama rasio % tetapi konsep berbeda → bandingkan perubahan pp / berdampingan.\n"
        "- **Indeks 100**: besaran positif dengan satuan berbeda/belum terkonfirmasi → indeks basis 100 atau % perubahan.\n"
        "- **Berdampingan**: selain itu → dua sumbu berlabel atau panel terpisah saja.")
    def_ids = [i for i in ["bank_umum::rasio_npl", "bpr::rasio_npl", "bank_umum::total_kredit_rp_m", "bank_umum::rasio_car",
                           "perusahaan_asuransi_jiwa::total_aset_asuransi_jiwa",
                           "perusahaan_asuransi_umum::total_aset_asuransi_umum", "perusahaan_asuransi_jiwa::rbc_asuransi_jiwa",
                           "pasar_modal::ihsg", "pasar_modal::kapitalisasi_pasar_rp_triliun", "dana_pensiun::total_investasi"]
               if i in set(ds.catalog["indicator_id"])]
    sel = st.multiselect("Indikator", ds.catalog["indicator_id"].tolist(), default=def_ids,
                         format_func=lambda i: f"{ds.info(i)['label_tampil']} — {ds.info(i)['sheet']}")
    if len(sel) >= 2:
        sers = {i: ds.series(i, gstart, gend) for i in sel}
        score, why = C.matrix(ds.catalog, sel, sers)
        t1, t2 = st.tabs(["📊 Matriks", "🧾 Data (grid)"])
        with t1:
            st.plotly_chart(viz.compat_heatmap(score, why), key="compat", config={"displaylogo": False})
        with t2:
            long = why.stack().reset_index()
            long.columns = ["indikator A", "indikator B", "keputusan & alasan"]
            st.dataframe(long[long["indikator A"] != long["indikator B"]], hide_index=True, height=420)
        st.caption("Konfirmasi satuan di sidebar akan memperbarui matriks (mis. Total Aset Asuransi Jiwa vs Umum menjadi "
                   "'Absolut' setelah keduanya dikonfirmasi bersatuan sama).")
    else:
        st.info("Pilih minimal dua indikator.")

# ---------------------------------------------------------------------------
# Tab 5 — Lintas sektor
# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("### Eksplorasi lintas sektor")
    st.caption("Hubungan di sini bersifat EKSPLORATIF (pergerakan bersama). Tidak menyatakan sebab-akibat atau transmisi "
               "risiko yang belum diuji.")
    presets = {
        "IHSG vs Total Investasi Dana Pensiun": ["pasar_modal::ihsg", "dana_pensiun::total_investasi"],
        "IHSG vs SBN & Obligasi Korporasi Dana Pensiun": ["pasar_modal::ihsg", "dana_pensiun::sbn", "dana_pensiun::obligasi_korporasi"],
        "NPL Bank Umum vs NPL BPR vs NPF Pembiayaan": ["bank_umum::rasio_npl", "bpr::rasio_npl", "perusahaan_pembiayaan::rasio_npf"],
        "Kapitalisasi Pasar vs Total Investasi Asuransi Jiwa": ["pasar_modal::kapitalisasi_pasar_rp_triliun",
                                                                "perusahaan_asuransi_jiwa::total_investasi_asuransi_jiwa"],
        "Pilih sendiri": [],
    }
    pc1, pc2 = st.columns([1, 2])
    preset = pc1.radio("Skenario", list(presets))
    valid = set(ds.catalog["indicator_id"])
    chosen = [i for i in presets[preset] if i in valid]
    if preset == "Pilih sendiri" or not chosen:
        chosen = pc2.multiselect("Indikator (2–6)", ds.catalog["indicator_id"].tolist(), max_selections=6,
                                 format_func=lambda i: f"{ds.info(i)['label_tampil']} — {ds.info(i)['sheet']}")
    else:
        pc2.markdown("**Indikator:** " + "; ".join(f"{ds.info(i)['label_tampil']} — {ds.info(i)['sheet']}" for i in chosen))
    if len(chosen) >= 2:
        res_x = engine.run(ds, manual_query(["korelasi", "bandingkan"]), chosen, gstart, gend, ss.th_override)
        render_result(res_x, "lintas")
    st.markdown("---")
    st.markdown("#### 🕸️ Network hubungan eksplisit")
    st.caption("Tepi hanya berasal dari aturan eksplisit di config/relations.json (identitas penjumlahan/rasio yang "
               "dirujuk dari deskripsi kamus) dan diwarnai menurut hasil uji pada data. Tepi korelasi opsional diberi "
               "gaya berbeda dan label eksploratif.")
    nc1, nc2 = st.columns([2, 1])
    nsheets = nc1.multiselect("Subsektor", sorted(ds.rule_results["sheet"].unique()),
                              default=sorted(ds.rule_results["sheet"].unique()))
    add_corr = nc2.toggle("Tambah tepi korelasi eksploratif (|r| perubahan ≥ 0,7)", value=False)
    corr_edges = []
    if add_corr and len(chosen) >= 2:
        for c in res_x.correlations:
            r = c.get("r_perubahan")
            if r is not None and abs(r) >= 0.7:
                corr_edges.append({"a": c["a"], "b": c["b"], "r": r, "n": c["n_perubahan"]})
        if not corr_edges:
            st.caption("Tidak ada pasangan dengan |r| perubahan ≥ 0,7 pada skenario terpilih.")
    elif add_corr:
        st.caption("Pilih skenario dengan ≥ 2 indikator untuk menghitung tepi korelasi.")
    nt1, nt2 = st.tabs(["🕸️ Network", "🧾 Data (grid)"])
    with nt1:
        st.plotly_chart(viz.network_chart(ds, nsheets, corr_edges), key="net", config={"displaylogo": False})
    with nt2:
        st.dataframe(ds.rule_results[ds.rule_results["sheet"].isin(nsheets)][["id", "sheet", "jenis", "target", "referensi",
                                                                               "status", "lolos_pct", "detail", "dasar"]],
                     hide_index=True)
        if corr_edges:
            st.dataframe(pd.DataFrame(corr_edges), hide_index=True)

# ---------------------------------------------------------------------------
# Tab 6 — Proyeksi
# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown("### Proyeksi sederhana")
    st.caption("Baseline yang mudah dijelaskan (naive, seasonal naive, drift, tren linier) dipilih via backtest "
               "rolling-origin. Perkiraan ilustratif — BUKAN prediksi resmi OJK.")
    f1, f2 = st.columns([3, 1])
    fid = f1.selectbox("Indikator", ds.catalog["indicator_id"].tolist(),
                       index=ds.catalog["indicator_id"].tolist().index("bpr::total_aset_rp_m")
                       if "bpr::total_aset_rp_m" in set(ds.catalog["indicator_id"]) else 0,
                       format_func=lambda i: f"{ds.info(i)['label_tampil']} — {ds.info(i)['sheet']}")
    hz = f2.radio("Horizon (bulan)", [3, 6, 12], index=1, horizontal=True)
    res_f = engine.run(ds, manual_query(["proyeksi"], hz), [fid], gstart, gend, ss.th_override, horizon=hz)
    render_result(res_f, "proyeksi")

# ---------------------------------------------------------------------------
# Tab 7 — Tata kelola
# ---------------------------------------------------------------------------
with tabs[6]:
    st.markdown("### Tata kelola & asal data")
    st.markdown(
        f"""
**Prinsip yang diterapkan POC ini**
1. **Kamus sebagai sumber kebenaran.** Katalog dibangun otomatis dari `Kamus_Data`; field di luar kamus diabaikan dan dilaporkan.
2. **Satuan tidak diasumsikan.** Field tanpa satuan di kamus berlabel *satuan belum terkonfirmasi*; konversi hanya via `config/unit_conversions.json` (saat ini {'kosong' if not json.loads((ROOT / 'config' / 'unit_conversions.json').read_text())['konversi'] else 'terisi'}).
3. **Stok ≠ arus.** Nilai stok/rasio/indeks tidak pernah dijumlahkan antarbulan; seri terindikasi kumulatif tahun berjalan ditandai.
4. **LLM opsional & terbatas.** Tanpa API key aplikasi berjalan penuh. Bila diaktifkan, LLM hanya menafsirkan pertanyaan (tanpa nilai data); seluruh angka dihitung kode.
5. **Threshold berlabel.** Semua threshold default adalah *threshold simulasi*; bukan batas regulasi OJK.
6. **Hubungan bukan kausalitas.** Korelasi diberi label eksploratif; network hanya dari aturan eksplisit.
7. **Proyeksi bukan prediksi resmi.**
8. **Data internal tidak masuk repositori/demo publik.** Repo memuat data statistik publik (`data/public/`) dan data simulasi; data internal hanya di `data/private/` (diabaikan git). Set `OASIS_PUBLIC_DEMO=1` pada deployment publik.
"""
    )
    st.markdown("**Asal data aktif**")
    st.json(ds.provenance)
    st.markdown("**Threshold aktif (default)**")
    st.dataframe(pd.DataFrame(load_thresholds()), hide_index=True)
    if ss.th_override:
        st.markdown("**Threshold konfigurasi pengguna**")
        st.dataframe(pd.DataFrame([{"indikator": k, **v[0]} for k, v in ss.th_override.items()]), hide_index=True)
    st.markdown("**Log audit sesi** (konfirmasi satuan, threshold, pilihan ambiguitas)")
    audit = pd.DataFrame(ss.audit, columns=["waktu", "aksi", "detail"])
    st.dataframe(audit, hide_index=True)
    st.download_button("Unduh log audit + konfirmasi satuan (JSON)",
                       json.dumps({"provenance": ds.provenance, "konfirmasi_satuan": ss.unit_conf,
                                   "threshold_pengguna": ss.th_override, "audit": ss.audit}, ensure_ascii=False,
                                  indent=2).encode("utf-8"), file_name="oasis_audit.json")
