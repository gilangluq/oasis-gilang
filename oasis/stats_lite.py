"""Statistik ringan tanpa scipy (numpy + math murni).

Dibuat agar OASIS tetap berjalan di lingkungan yang memblokir DLL scipy
(mis. kebijakan Application Control di laptop kantor). Hasil identik dengan
scipy.stats.linregress / pearsonr / spearmanr untuk kasus yang dipakai OASIS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction untuk incomplete beta (Numerical Recipes)."""
    max_iter, eps, fpmin = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > fpmin else fpmin)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > fpmin else fpmin)
        c = 1.0 + aa / c
        c = c if abs(c) > fpmin else fpmin
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > fpmin else fpmin)
        c = 1.0 + aa / c
        c = c if abs(c) > fpmin else fpmin
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    front = math.exp(lbeta)
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def t_two_sided_p(t: float, df: int) -> float:
    if df <= 0 or not np.isfinite(t):
        return float("nan") if not np.isfinite(t) else 0.0
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def _r_pvalue(r: float, n: int) -> float:
    if n <= 2:
        return float("nan")
    r = max(min(r, 1.0), -1.0)
    if abs(r) == 1.0:
        return 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return t_two_sided_p(t, n - 2)


def pearsonr(x, y) -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    xm, ym = x - x.mean(), y - y.mean()
    den = math.sqrt(float((xm ** 2).sum() * (ym ** 2).sum()))
    r = float((xm * ym).sum() / den) if den else float("nan")
    return r, _r_pvalue(r, len(x))


def _rank(a: np.ndarray) -> np.ndarray:
    """Peringkat rata-rata untuk nilai sama (seperti scipy.stats.rankdata)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def spearmanr(x, y) -> tuple[float, float]:
    return pearsonr(_rank(np.asarray(x, float)), _rank(np.asarray(y, float)))


@dataclass
class LinregressResult:
    slope: float
    intercept: float
    rvalue: float
    pvalue: float
    stderr: float


def linregress(x, y) -> LinregressResult:
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    sxy = float(((x - xm) * (y - ym)).sum())
    syy = float(((y - ym) ** 2).sum())
    slope = sxy / sxx if sxx else float("nan")
    intercept = ym - slope * xm
    r = sxy / math.sqrt(sxx * syy) if sxx and syy else 0.0
    df = n - 2
    if df > 0 and abs(r) < 1:
        stderr = math.sqrt((1 - r * r) * syy / sxx / df)
        p = t_two_sided_p(slope / stderr, df) if stderr else 0.0
    else:
        stderr, p = 0.0, (0.0 if df > 0 else float("nan"))
    return LinregressResult(slope, intercept, r, p, stderr)
