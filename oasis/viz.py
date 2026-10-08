"""Visualisasi Plotly OASIS.

Prinsip:
* Warna seri tetap per indikator dalam satu hasil (bukan per peringkat).
* Merah DICADANGKAN untuk penanda anomali; tidak dipakai sebagai warna seri.
* Tooltip memuat periode, nilai+satuan, perubahan dari periode sebelumnya, dan YoY.
* Dua sumbu hanya untuk satuan berbeda, dengan judul sumbu bersatuan dan catatan
  bahwa tinggi garis antarsumbu tidak dapat dibandingkan.
* Threshold digambar putus-putus dengan label sumbernya.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import analytics as A
from .engine import AnalysisResult

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4", "#008300"]
ANOMALY_COLOR = "#d62728"
THRESHOLD_COLOR = "#6b6a66"
FORECAST_BAND = "rgba(42,120,214,0.15)"
FONT = dict(family="Inter, Segoe UI, Arial, sans-serif", size=13, color="#2b2b2b")


def color_for(res: AnalysisResult, iid: str) -> str:
    return SERIES_COLORS[res.ids.index(iid) % len(SERIES_COLORS)]


def _base_layout(fig: go.Figure, title: str, height: int = 440) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)),
        template="plotly_white", font=FONT, height=height, hovermode="closest", separators=",.",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0, font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=False, ticks="outside")
    fig.update_yaxes(gridcolor="#ececea", tickformat=",.2~f", exponentformat="none")
    return fig


def _chg_txt(v, mode) -> str:
    if v is None or pd.isna(v):
        return "–"
    sfx = {"pp": " pp", "pct": "%", "abs": ""}[mode]
    return f"{'+' if v >= 0 else ''}{A.fmt_num(float(v))}{sfx}"


def _custom(res: AnalysisResult, iid: str, idx: pd.DatetimeIndex) -> np.ndarray:
    info = res.infos[iid]
    ch = res.changes[iid].reindex(idx)
    mode = res.changes[iid].attrs.get("mode", "abs")
    rows = []
    for p in idx:
        r = ch.loc[p]
        rows.append([A.period_label(p), A.fmt_val(r["nilai"], info), _chg_txt(r["mom"], mode),
                     _chg_txt(r["yoy"], mode), info["label_tampil"], info["sheet"]])
    return np.array(rows, dtype=object)


HOVER = ("<b>%{customdata[4]}</b> — %{customdata[5]}<br>Periode: %{customdata[0]}<br>Nilai: %{customdata[1]}"
         "<br>vs bulan sebelumnya: %{customdata[2]}<br>YoY: %{customdata[3]}<extra></extra>")


def _add_anomalies(fig, res, iid, y_map=None, secondary=None, row=None, col=None):
    an = res.anomalies.get(iid)
    if an is None or an.empty:
        return
    info = res.infos[iid]
    ys = an["nilai"] if y_map is None else an["periode"].map(y_map)
    cd = np.array([[A.period_label(p), A.fmt_val(v, info), reason, info["label_tampil"], info["sheet"], iid,
                    pd.Timestamp(p).strftime("%Y-%m-%d")]
                   for p, v, reason in zip(an["periode"], an["nilai"], an["alasan"])], dtype=object)
    tr = go.Scatter(x=an["periode"], y=ys, mode="markers", name="Anomali (klik)",
                    marker=dict(color=ANOMALY_COLOR, size=12, line=dict(color="white", width=2)),
                    customdata=cd, legendgroup="anomali", showlegend=not any(t.name == "Anomali (klik)" for t in fig.data),
                    hovertemplate="<b>Anomali</b> — %{customdata[3]} (%{customdata[4]})<br>%{customdata[0]}: "
                                  "%{customdata[1]}<br>%{customdata[2]}<extra></extra>")
    if secondary is not None:
        fig.add_trace(tr, secondary_y=secondary)
    elif row is not None:
        fig.add_trace(tr, row=row, col=col)
    else:
        fig.add_trace(tr)


def _add_thresholds(fig, res, iid, secondary=None, row=None, col=None):
    """Threshold sebagai garis putus-putus berlabel (trace, agar tampil benar juga di dua sumbu/panel)."""
    s = res.series.get(iid)
    if s is None or s.empty:
        return
    x0, x1 = s.index.min(), s.index.max()
    fc = res.forecasts.get(iid)
    if fc is not None and fc.ok:
        x1 = fc.prediksi.index.max()
    for h in res.thresholds.get(iid, []):
        t = h["threshold"]
        tr = go.Scatter(x=[x0, x1], y=[t.nilai, t.nilai], mode="lines", name=f"{t.label} — {t.sumber}",
                        line=dict(color=THRESHOLD_COLOR, dash="dash", width=1.5),
                        hovertemplate=f"{t.label}<br>Sumber: {t.sumber}<extra></extra>")
        if secondary is not None:
            fig.add_trace(tr, secondary_y=secondary)
        elif row is not None:
            tr.showlegend = False
            fig.add_trace(tr, row=row, col=col)
        else:
            fig.add_trace(tr)


def _axis_title(info: dict) -> str:
    u = A.unit_text(info)
    lab = info["label_tampil"]
    return lab if f"({u})".casefold() in lab.casefold() else f"{lab} ({u})"


# ---------------------------------------------------------------------------
def line_chart(res: AnalysisResult, transform: str = "nilai") -> go.Figure:
    """transform: 'nilai' (sumbu sama) | 'indeks' (basis 100 pada periode irisan pertama) | 'yoy'."""
    fig = go.Figure()
    ids = [i for i in res.ids if not res.series[i].empty]
    title_sfx = ""
    if transform == "indeks" and len(ids) > 1:
        al = res.aligned["irisan"]
        idx_df, base = A.index_base100(al)
        for iid in ids:
            s = idx_df[iid]
            cd = _custom(res, iid, s.index)
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=res.label(iid),
                                     line=dict(color=color_for(res, iid), width=2), customdata=cd,
                                     hovertemplate=HOVER.replace("Nilai:", "Indeks: " + "%{y:.1f}<br>Nilai:")))
            _add_anomalies(fig, res, iid, y_map=s.to_dict())
        fig.update_yaxes(title_text=f"Indeks (basis 100 = {A.period_label(base)})")
        title_sfx = f" — indeks basis 100 ({A.period_label(base)}), periode irisan"
    elif transform == "yoy":
        for iid in ids:
            ch = res.changes[iid]
            s = ch["yoy"].dropna()
            mode = ch.attrs.get("mode")
            cd = _custom(res, iid, s.index)
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=f"{res.label(iid)} "
                                     f"[{'pp' if mode == 'pp' else '%' if mode == 'pct' else 'selisih'}]",
                                     line=dict(color=color_for(res, iid), width=2), customdata=cd, hovertemplate=HOVER))
        modes = {res.changes[i].attrs.get("mode") for i in ids}
        fig.update_yaxes(title_text="Perubahan YoY (" + ("pp" if modes == {"pp"} else "%" if modes == {"pct"}
                                                         else "satuan campuran — lihat legenda") + ")")
        fig.add_hline(y=0, line=dict(color="#999", width=1))
        title_sfx = " — perubahan YoY"
    else:
        for iid in ids:
            s = res.series[iid]
            cd = _custom(res, iid, s.index)
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=res.label(iid),
                                     line=dict(color=color_for(res, iid), width=2), customdata=cd, hovertemplate=HOVER))
            _add_anomalies(fig, res, iid)
            if len(ids) == 1:
                _add_thresholds(fig, res, iid)
        units = {A.unit_text(res.infos[i]) for i in ids}
        fig.update_yaxes(title_text=_axis_title(res.infos[ids[0]]) if len(ids) == 1 else f"Nilai ({', '.join(units)})")
    title = (res.label(ids[0]) if len(ids) == 1 else "Perbandingan " + ", ".join(res.label(i) for i in ids)) + title_sfx
    return _base_layout(fig, title)


def dual_axis(res: AnalysisResult) -> go.Figure:
    ids = [i for i in res.ids if not res.series[i].empty][:2]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for k, iid in enumerate(ids):
        s = res.series[iid]
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=f"{res.label(iid)} ({'kiri' if k == 0 else 'kanan'})",
                                 line=dict(color=color_for(res, iid), width=2, dash="solid" if k == 0 else "dot"),
                                 customdata=_custom(res, iid, s.index), hovertemplate=HOVER), secondary_y=(k == 1))
        _add_anomalies(fig, res, iid, secondary=(k == 1))
        _add_thresholds(fig, res, iid, secondary=(k == 1))
        fig.update_yaxes(title_text=_axis_title(res.infos[iid]), secondary_y=(k == 1),
                         title_font=dict(color=color_for(res, iid)), tickfont=dict(color=color_for(res, iid)))
    fig = _base_layout(fig, "Dua sumbu: " + " vs ".join(res.label(i) for i in ids), height=480)
    fig.update_yaxes(secondary_y=True, gridcolor="rgba(0,0,0,0)")
    return fig


def small_multiples(res: AnalysisResult) -> go.Figure:
    ids = [i for i in res.ids if not res.series[i].empty]
    n = len(ids)
    cols = 1 if n <= 2 else 2
    rows = math.ceil(n / cols)
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, vertical_spacing=0.08 if rows < 5 else 0.04,
                        subplot_titles=[f"{res.label(i)} ({A.unit_text(res.infos[i])})" for i in ids])
    for k, iid in enumerate(ids):
        r, c = k // cols + 1, k % cols + 1
        s = res.series[iid]
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=res.label(iid), showlegend=False,
                                 line=dict(color=color_for(res, iid), width=2), customdata=_custom(res, iid, s.index),
                                 hovertemplate=HOVER), row=r, col=c)
        _add_anomalies(fig, res, iid, row=r, col=c)
        _add_thresholds(fig, res, iid, row=r, col=c)
    fig = _base_layout(fig, "Berdampingan — tiap panel memiliki sumbu & satuannya sendiri", height=max(320, 230 * rows))
    fig.update_annotations(font_size=12)
    return fig


def bar_single(res: AnalysisResult, kind: str = "nilai") -> go.Figure:
    iid = [i for i in res.ids if not res.series[i].empty][0]
    info = res.infos[iid]
    ch = res.changes[iid]
    mode = ch.attrs.get("mode", "abs")
    col = {"nilai": "nilai", "yoy": "yoy", "mom": "mom"}[kind]
    s = ch[col].dropna()
    colors = [color_for(res, iid) if (kind == "nilai" or v >= 0) else "#9a9893" for v in s.values]
    fig = go.Figure(go.Bar(x=s.index, y=s.values, marker=dict(color=colors, cornerradius=4), name=res.label(iid),
                           customdata=_custom(res, iid, s.index), hovertemplate=HOVER))
    if kind != "nilai":
        fig.add_hline(y=0, line=dict(color="#999", width=1))
        ylab = f"Perubahan {kind.upper()} ({'pp' if mode == 'pp' else '%' if mode == 'pct' else 'selisih, ' + A.unit_text(info)})"
    else:
        ylab = _axis_title(info)
        _add_thresholds(fig, res, iid)
    fig.update_yaxes(title_text=ylab)
    fig.update_layout(bargap=0.25)
    return _base_layout(fig, f"{res.label(iid)} — {'nilai bulanan' if kind == 'nilai' else 'perubahan ' + kind.upper()}")


def bar_last(res: AnalysisResult) -> go.Figure:
    ids = [i for i in res.ids if not res.series[i].empty]
    al = res.aligned["irisan"]
    p = al.index.max()
    vals = [float(al.loc[p, i]) for i in ids]
    fig = go.Figure(go.Bar(x=[res.label(i) for i in ids], y=vals, marker=dict(color=[color_for(res, i) for i in ids],
                                                                                cornerradius=4),
                           text=[A.fmt_val(v, res.infos[i]) for v, i in zip(vals, ids)], textposition="outside",
                           hovertemplate="%{x}<br>%{text}<extra></extra>"))
    fig.update_yaxes(title_text=f"Nilai ({A.unit_text(res.infos[ids[0]])})")
    return _base_layout(fig, f"Nilai pada periode irisan terakhir ({A.period_label(p)})")


def scatter(res: AnalysisResult) -> go.Figure:
    a, b = [i for i in res.ids if not res.series[i].empty][:2]
    df = pd.concat([res.series[a].rename("a"), res.series[b].rename("b")], axis=1).dropna()
    ia, ib = res.infos[a], res.infos[b]
    cd = np.array([[A.period_label(p), A.fmt_val(x, ia), A.fmt_val(y, ib)] for p, x, y in zip(df.index, df["a"], df["b"])],
                  dtype=object)
    t = (df.index.year + (df.index.month - 1) / 12).values
    fig = go.Figure(go.Scatter(x=df["a"], y=df["b"], mode="markers",
                               marker=dict(size=9, color=t, colorscale=[[0, "#c9dcf5"], [1, "#1c4f8f"]], showscale=True,
                                           colorbar=dict(title="Tahun", thickness=10), line=dict(color="white", width=1)),
                               customdata=cd, name="Periode",
                               hovertemplate="%{customdata[0]}<br>" + ia["label_tampil"] + ": %{customdata[1]}<br>"
                                             + ib["label_tampil"] + ": %{customdata[2]}<extra></extra>"))
    if len(df) >= 6:
        k, c0 = np.polyfit(df["a"], df["b"], 1)
        xs = np.linspace(df["a"].min(), df["a"].max(), 20)
        fig.add_trace(go.Scatter(x=xs, y=k * xs + c0, mode="lines", line=dict(color="#6b6a66", dash="dot", width=1.5),
                                 name="Garis OLS (deskriptif)", hoverinfo="skip"))
    fig.update_xaxes(title_text=_axis_title(ia), showgrid=True, gridcolor="#ececea")
    fig.update_yaxes(title_text=_axis_title(ib))
    return _base_layout(fig, f"Scatter eksploratif: {res.label(a)} vs {res.label(b)}", height=480)


def corr_heatmap(res: AnalysisResult, basis: str = "perubahan") -> tuple[go.Figure, pd.DataFrame]:
    ids = [i for i in res.ids if not res.series[i].empty]
    if basis == "perubahan":
        data = {}
        for i in ids:
            ch = res.changes[i]
            data[res.label(i)] = ch["mom"]
        df = pd.DataFrame(data)
        sub = "perubahan bulanan (MoM: pp untuk rasio, % untuk besaran)"
    else:
        df = pd.DataFrame({res.label(i): res.series[i] for i in ids})
        sub = "level (rawan korelasi semu bila sama-sama bertren)"
    corr = df.corr(min_periods=12)
    n = df.notna().astype(int).T.dot(df.notna().astype(int))
    text = [[f"{corr.iloc[r, c]:.2f}<br>n={n.iloc[r, c]}" if pd.notna(corr.iloc[r, c]) else "n<12"
             for c in range(len(corr))] for r in range(len(corr))]
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, text=text,
                               texttemplate="%{text}", colorscale=[[0, "#c4471f"], [0.5, "#f1f0ec"], [1, "#1c5fae"]],
                               colorbar=dict(title="r", thickness=10),
                               hovertemplate="%{y}<br>%{x}<br>r Pearson = %{z:.2f}<extra></extra>"))
    fig = _base_layout(fig, f"Heatmap korelasi eksploratif — {sub}", height=max(420, 70 * len(ids) + 160))
    fig.update_xaxes(tickangle=-30)
    return fig, corr.round(3)


def forecast_chart(res: AnalysisResult, iid: str) -> go.Figure:
    fr = res.forecasts[iid]
    info = res.infos[iid]
    s = res.series[iid]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="Historis", line=dict(color=color_for(res, iid), width=2),
                             customdata=_custom(res, iid, s.index), hovertemplate=HOVER))
    _add_anomalies(fig, res, iid)
    if fr.ok:
        p = fr.prediksi
        x_band = list(p.index) + list(p.index[::-1])
        y_band = list(p["atas"]) + list(p["bawah"][::-1])
        fig.add_trace(go.Scatter(x=x_band, y=y_band, fill="toself", fillcolor=FORECAST_BAND, line=dict(width=0),
                                 name="Interval perkiraan ~80%", hoverinfo="skip"))
        xs = [s.index[-1]] + list(p.index)
        ys = [float(s.iloc[-1])] + list(p["prediksi"])
        cd = [["", "", "", ""]] + [[A.period_label(d), A.fmt_val(v, info), A.fmt_val(lo, info), A.fmt_val(hi, info)]
                                   for d, v, lo, hi in zip(p.index, p["prediksi"], p["bawah"], p["atas"])]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=f"Proyeksi — {fr.metode_label}",
                                 line=dict(color=color_for(res, iid), width=2, dash="dash"), marker=dict(size=7),
                                 customdata=np.array(cd, dtype=object),
                                 hovertemplate="<b>Perkiraan</b> %{customdata[0]}<br>%{customdata[1]}<br>Interval: "
                                               "%{customdata[2]} – %{customdata[3]}<extra></extra>"))
        fig.add_vrect(x0=p.index[0] - pd.offsets.MonthEnd(1), x1=p.index[-1], fillcolor="#f3f2ee", opacity=0.5,
                      line_width=0, layer="below", annotation_text="Perkiraan (bukan prediksi resmi OJK)",
                      annotation_position="top left", annotation_font=dict(size=11, color="#6b6a66"))
    _add_thresholds(fig, res, iid)
    fig.update_yaxes(title_text=_axis_title(info))
    title = f"Proyeksi {fr.horizon} bulan: {res.label(iid)}" if fr.ok else f"{res.label(iid)} — proyeksi tidak dibuat"
    return _base_layout(fig, title, height=460)


def compat_heatmap(score: pd.DataFrame, why: pd.DataFrame) -> go.Figure:
    labels = {3: "Absolut", 2: "Perubahan pp", 1: "Indeks 100", 0: "Berdampingan"}
    text = [[labels.get(int(v), "") for v in row] for row in score.values]
    fig = go.Figure(go.Heatmap(
        z=score.values, x=score.columns, y=score.index, zmin=0, zmax=3, text=text, texttemplate="%{text}",
        customdata=why.values, colorscale=[[0, "#e9e7e1"], [0.33, "#c9dcf5"], [0.66, "#7fb0ea"], [1, "#1c5fae"]],
        showscale=False, hovertemplate="%{y}<br>vs %{x}<br>%{customdata}<extra></extra>"))
    fig = _base_layout(fig, "Matriks kompatibilitas perbandingan", height=max(420, 34 * len(score) + 200))
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10), autorange="reversed")
    return fig


def network_chart(ds, sheets: list[str] | None = None, corr_edges: list[dict] | None = None) -> go.Figure:
    """Graph hanya dari aturan eksplisit (config/relations.json) + (opsional) tepi korelasi berlabel jelas."""
    rr = ds.rule_results
    if sheets:
        rr = rr[rr["sheet"].isin(sheets)]
    cat = ds.catalog.set_index("indicator_id")
    nodes, edges = {}, []
    for _, r in rr.iterrows():
        tgt = r["target_id"]
        for ref in r["referensi_id"]:
            if tgt and ref:
                nodes[tgt] = r["sheet"]
                nodes[ref] = r["sheet"]
                edges.append((ref, tgt, r["id"], r["jenis"], r["status"], r["detail"], r["dasar"]))
    for e in corr_edges or []:
        nodes.setdefault(e["a"], cat.loc[e["a"], "sheet"])
        nodes.setdefault(e["b"], cat.loc[e["b"], "sheet"])
        edges.append((e["a"], e["b"], "korelasi", "korelasi eksploratif", "eksploratif",
                      f"r perubahan = {e['r']:.2f} (n={e['n']})", "Data: korelasi perubahan bulanan; bukan kausal."))
    fig = go.Figure()
    if not nodes:
        fig.add_annotation(text="Tidak ada hubungan eksplisit untuk subsektor terpilih.", showarrow=False)
        return _base_layout(fig, "Network hubungan eksplisit")
    sheets_present = list(dict.fromkeys(nodes.values()))
    pos = {}
    for si, sh in enumerate(sheets_present):
        members = [n for n, s in nodes.items() if s == sh]
        cx, cy = (si % 3) * 3.2, -(si // 3) * 3.0
        for k, n in enumerate(members):
            ang = 2 * math.pi * k / max(1, len(members))
            pos[n] = (cx + math.cos(ang), cy + math.sin(ang))
    st_color = {"lolos": "#1baf7a", "sebagian": "#eda100", "gagal": ANOMALY_COLOR, "tidak dapat diuji": "#a8a7a2",
                "eksploratif": "#4a3aa7"}
    for (a, b, rid, jenis, status, detail, dasar) in edges:
        (x0, y0), (x1, y1) = pos[a], pos[b]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color=st_color.get(status, "#999"), width=2.5,
                                           dash="dot" if status in ("tidak dapat diuji", "eksploratif") else "solid")))
        fig.add_trace(go.Scatter(x=[(x0 + x1) / 2], y=[(y0 + y1) / 2], mode="markers", showlegend=False,
                                 marker=dict(size=14, color=st_color.get(status, "#999"), opacity=0.01),
                                 hovertemplate=f"<b>{rid}</b> ({jenis}) — {status}<br>{detail}<br>Dasar: {dasar}<extra></extra>"))
    for status, colr in st_color.items():
        if any(e[4] == status for e in edges):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=colr, width=3), name=f"Tepi: {status}"))
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    labels = [cat.loc[n, "label_tampil"] for n in nodes]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=labels, textposition="bottom center",
                             textfont=dict(size=10), name="Indikator",
                             marker=dict(size=16, color=[SERIES_COLORS[sheets_present.index(nodes[n]) % 7] for n in nodes],
                                         line=dict(color="white", width=2)),
                             customdata=[[nodes[n]] for n in nodes],
                             hovertemplate="%{text}<br>%{customdata[0]}<extra></extra>"))
    for si, sh in enumerate(sheets_present):
        fig.add_annotation(x=(si % 3) * 3.2, y=-(si // 3) * 3.0 + 1.55, text=f"<b>{sh}</b>", showarrow=False,
                           font=dict(size=12))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    return _base_layout(fig, "Network: hanya hubungan dari aturan eksplisit (warna = hasil uji data)",
                        height=max(420, 330 * math.ceil(len(sheets_present) / 3)))
