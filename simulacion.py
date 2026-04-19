"""
simulacion.py
=============
Simulación de eventos discretos con SimPy.
Incluye: proceso de lotes, sensores virtuales, cálculo de KPIs y utilización.
"""

import math
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import simpy
from datos import PRODUCTOS, RUTAS, TAMANO_LOTE_BASE, CAPACIDAD_BASE, DEM_HISTORICA, PROD_COLORS


# ── Motor de simulación ───────────────────────────────────────────────────────

def run_simulacion(
    plan_unidades: dict,
    cap_recursos: dict = None,
    falla: bool = False,
    factor_t: float = 1.0,
    tamano_lote: dict = None,
    semilla: int = 42,
):
    """
    Ejecuta la simulación de eventos discretos.

    Parámetros
    ----------
    plan_unidades : dict   {producto: unidades_mes}
    cap_recursos  : dict   capacidad de cada recurso (defecto CAPACIDAD_BASE)
    falla         : bool   simula falla en horno (+10-30 min por operación)
    factor_t      : float  escala tiempos de proceso (0.8 = doble turno)
    tamano_lote   : dict   tamaño de lote por producto
    semilla       : int    semilla aleatoria para reproducibilidad

    Retorna
    -------
    df_lotes : pd.DataFrame   registro por lote
    df_uso   : pd.DataFrame   uso de recursos en el tiempo
    df_sens  : pd.DataFrame   lecturas del sensor de horno
    """
    random.seed(semilla)
    np.random.seed(semilla)
    if cap_recursos is None:
        cap_recursos = CAPACIDAD_BASE.copy()
    if tamano_lote is None:
        tamano_lote = TAMANO_LOTE_BASE.copy()

    lotes_data, uso_rec, sensores = [], [], []

    def reg_uso(env, recursos):
        ts = round(env.now, 3)
        for nm, r in recursos.items():
            uso_rec.append({
                "tiempo": ts, "recurso": nm,
                "ocupados": r.count, "cola": len(r.queue), "capacidad": r.capacity,
            })

    def sensor_horno(env, recursos):
        while True:
            ocp  = recursos["horno"].count
            temp = round(np.random.normal(160 + ocp * 20, 5), 2)
            sensores.append({
                "tiempo": round(env.now, 1), "temperatura": temp,
                "horno_ocup": ocp, "horno_cola": len(recursos["horno"].queue),
            })
            yield env.timeout(10)

    def proceso_lote(env, lid, prod, tam, recursos):
        t0     = env.now
        esperas = {}
        for etapa, rec_nm, tmin, tmax in RUTAS[prod]:
            escala = math.sqrt(tam / TAMANO_LOTE_BASE[prod])
            tp     = random.uniform(tmin, tmax) * escala * factor_t
            if falla and rec_nm == "horno":
                tp += random.uniform(10, 30)
            reg_uso(env, recursos)
            t_ei = env.now
            with recursos[rec_nm].request() as req:
                yield req
                esperas[etapa] = round(env.now - t_ei, 3)
                reg_uso(env, recursos)
                yield env.timeout(tp)
            reg_uso(env, recursos)
        lotes_data.append({
            "lote_id":       lid,
            "producto":      prod,
            "tamano":        tam,
            "t_creacion":    round(t0, 3),
            "t_fin":         round(env.now, 3),
            "tiempo_sistema":round(env.now - t0, 3),
            "total_espera":  round(sum(esperas.values()), 3),
        })

    env      = simpy.Environment()
    recursos = {nm: simpy.Resource(env, capacity=cap) for nm, cap in cap_recursos.items()}
    env.process(sensor_horno(env, recursos))

    dur_mes = 44 * 4 * 60
    lotes   = []
    ctr     = [0]
    for prod, unid in plan_unidades.items():
        if unid <= 0:
            continue
        tam  = tamano_lote[prod]
        n    = math.ceil(unid / tam)
        tasa = dur_mes / max(n, 1)
        ta   = random.expovariate(1 / max(tasa, 1))
        rem  = unid
        for _ in range(n):
            lotes.append((round(ta, 2), prod, min(tam, int(rem))))
            rem -= tam
            ta  += random.expovariate(1 / max(tasa, 1))
    lotes.sort(key=lambda x: x[0])

    def lanzador():
        for ta, prod, tam in lotes:
            yield env.timeout(max(ta - env.now, 0))
            lid = f"{prod[:3].upper()}_{ctr[0]:04d}"
            ctr[0] += 1
            env.process(proceso_lote(env, lid, prod, tam, recursos))

    env.process(lanzador())
    env.run(until=dur_mes * 1.8)

    df_l = pd.DataFrame(lotes_data) if lotes_data else pd.DataFrame()
    df_u = pd.DataFrame(uso_rec)    if uso_rec    else pd.DataFrame()
    df_s = pd.DataFrame(sensores)   if sensores   else pd.DataFrame()
    return df_l, df_u, df_s


# ── Métricas ──────────────────────────────────────────────────────────────────

def calc_utilizacion(df_u: pd.DataFrame) -> pd.DataFrame:
    """Calcula % utilización, cola promedio/máxima y cuello de botella."""
    if df_u.empty:
        return pd.DataFrame()
    filas = []
    for rec, grp in df_u.groupby("recurso"):
        grp  = grp.sort_values("tiempo").reset_index(drop=True)
        cap  = grp["capacidad"].iloc[0]
        t    = grp["tiempo"].values
        ocp  = grp["ocupados"].values
        if len(t) > 1 and (t[-1] - t[0]) > 0:
            fn   = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            area = fn(ocp, t)
            util = round(area / (cap * (t[-1] - t[0])) * 100, 2)
        else:
            util = 0.0
        filas.append({
            "Recurso":       rec,
            "Utilización_%": util,
            "Cola Prom":     round(grp["cola"].mean(), 3),
            "Cola Máx":      int(grp["cola"].max()),
            "Capacidad":     int(cap),
            "Cuello Botella":util >= 80 or grp["cola"].mean() > 0.5,
        })
    return pd.DataFrame(filas).sort_values("Utilización_%", ascending=False).reset_index(drop=True)


def calc_kpis(df_l: pd.DataFrame, plan: dict) -> pd.DataFrame:
    """KPIs de producción: throughput, cycle time, lead time, WIP, takt, cumplimiento."""
    if df_l.empty:
        return pd.DataFrame()
    dur  = (df_l["t_fin"].max() - df_l["t_creacion"].min()) / 60
    rows = []
    for p in PRODUCTOS:
        sub = df_l[df_l["producto"] == p]
        if sub.empty:
            continue
        und      = sub["tamano"].sum()
        plan_und = plan.get(p, 0)
        tp       = round(und / max(dur, 0.01), 3)
        ct       = round((sub["tiempo_sistema"] / sub["tamano"]).mean(), 3)
        lt       = round(sub["tiempo_sistema"].mean(), 3)
        dem_avg  = sum(DEM_HISTORICA[p]) / 12
        takt     = round((44 * 4 * 60) / max(dem_avg / TAMANO_LOTE_BASE[p], 1), 2)
        wip      = round(tp * (lt / 60), 2)
        rows.append({
            "Producto":             p,
            "Und Producidas":       und,
            "Plan":                 plan_und,
            "Throughput (und/h)":   tp,
            "Cycle Time (min/und)": ct,
            "Lead Time (min/lote)": lt,
            "WIP Prom":             wip,
            "Takt Time (min/lote)": takt,
            "Cumplimiento %":       round(min(und / max(plan_und, 1) * 100, 100), 2),
        })
    return pd.DataFrame(rows)


# ── Visualizaciones ───────────────────────────────────────────────────────────

def fig_gantt(df_l: pd.DataFrame, theme: dict, n: int = 80) -> go.Figure:
    """Diagrama de Gantt de lotes de producción."""
    if df_l.empty:
        return go.Figure()
    sub = df_l.head(n).copy().reset_index(drop=True)
    fig = go.Figure()
    for _, row in sub.iterrows():
        col = PROD_COLORS.get(row["producto"], "#6366F1")
        fig.add_trace(go.Bar(
            x=[row["tiempo_sistema"]], y=[row["lote_id"]],
            base=[row["t_creacion"]], orientation="h",
            marker_color=col, marker_line_width=0, opacity=0.82,
            hovertemplate=(
                f"<b>{row['producto'].replace('_',' ')}</b><br>"
                f"Lote: {row['lote_id']}<br>"
                f"Inicio: {row['t_creacion']:.0f} min<br>"
                f"Duración: {row['tiempo_sistema']:.1f} min<extra></extra>"
            ),
            showlegend=False,
        ))
    for p, c in PROD_COLORS.items():
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=c,
                             name=p.replace("_", " "), showlegend=True))
    fig.update_layout(
        **theme,
        barmode="overlay",
        height=max(380, len(sub) * 8),
        title=dict(text="Diagrama de Gantt — lotes de producción", x=0.5),
        xaxis_title="Tiempo simulado (min)",
        yaxis_title="Lote",
        yaxis=dict(showticklabels=len(sub) <= 40),
    )
    return fig


def fig_colas(df_u: pd.DataFrame, theme: dict) -> go.Figure:
    """Evolución de colas por recurso."""
    if df_u.empty:
        return go.Figure()
    paleta = ["#6366F1", "#0EA5E9", "#10B981", "#F59E0B", "#EC4899", "#A78BFA"]
    fig    = go.Figure()
    for i, (rec, grp) in enumerate(df_u.groupby("recurso")):
        grp = grp.sort_values("tiempo")
        col = paleta[i % len(paleta)]
        r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
        fig.add_trace(go.Scatter(
            x=grp["tiempo"], y=grp["cola"],
            mode="lines", name=rec,
            line=dict(color=col, width=2),
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.05)",
            hovertemplate=f"<b>{rec}</b><br>t=%{{x:.0f}}<br>Cola=%{{y}}<extra></extra>",
        ))
    fig.update_layout(
        **theme, height=380,
        title=dict(text="Evolución de colas por recurso", x=0.5),
        xaxis_title="Tiempo simulado (min)",
        yaxis_title="Tamaño de cola",
    )
    return fig


def fig_utilizacion(df_u: pd.DataFrame, theme: dict) -> go.Figure:
    """Barras horizontales de utilización y cola promedio."""
    df_ut = calc_utilizacion(df_u)
    if df_ut.empty:
        return go.Figure()
    C_RED   = "#EF4444"
    C_AMBER = "#F59E0B"
    C_ACCENT= "#6366F1"

    colores = [
        C_RED if u >= 80 else C_AMBER if u >= 60 else C_ACCENT
        for u in df_ut["Utilización_%"]
    ]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Utilización por recurso (%)", "Cola promedio"])
    fig.add_trace(go.Bar(
        y=df_ut["Recurso"], x=df_ut["Utilización_%"],
        orientation="h", marker_color=colores, marker_line_width=0,
        text=df_ut["Utilización_%"].apply(lambda v: f"{v:.0f}%"),
        textposition="outside", showlegend=False,
        hovertemplate="%{y}<br><b>%{x:.1f}%</b><extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        y=df_ut["Recurso"], x=df_ut["Cola Prom"],
        orientation="h", marker_color="#A78BFA", marker_line_width=0,
        text=df_ut["Cola Prom"].apply(lambda v: f"{v:.2f}"),
        textposition="outside", showlegend=False,
    ), row=1, col=2)
    fig.add_vline(x=80, line_dash="dash", line_color=C_RED,
                  annotation_text="⚠ 80%", annotation_font_color=C_RED, row=1, col=1)
    fig.update_layout(
        **theme, height=360,
        title=dict(text="Utilización de recursos — detección de cuellos de botella", x=0.5),
        margin=dict(l=85, r=70, t=55, b=40),
    )
    return fig


def fig_kpis_radar(df_kpi: pd.DataFrame, theme: dict) -> go.Figure:
    """Radar normalizado de KPIs por producto."""
    if df_kpi.empty:
        return go.Figure()
    cats = [
        "Throughput (und/h)", "Cycle Time (min/und)",
        "Lead Time (min/lote)", "WIP Prom", "Cumplimiento %",
    ]
    fig = go.Figure()
    for _, row in df_kpi.iterrows():
        vals  = [row.get(c, 0) for c in cats]
        maxv  = [max(df_kpi[c].max(), 0.01) for c in cats]
        norm  = [round(v / m * 100, 1) for v, m in zip(vals, maxv)]
        norm.append(norm[0])
        fig.add_trace(go.Scatterpolar(
            r=norm, theta=cats + [cats[0]],
            name=row["Producto"].replace("_", " "),
            fill="toself", opacity=0.5,
            line=dict(color=PROD_COLORS.get(row["Producto"], "#6366F1"), width=2.5),
        ))
    fig.update_layout(
        **theme, height=420,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=10)),
        ),
        title=dict(text="Radar de KPIs por producto (normalizado)", x=0.5),
    )
    return fig


def fig_sensores(df_s: pd.DataFrame, theme: dict) -> go.Figure:
    """Monitor del horno: temperatura y ocupación."""
    if df_s.empty:
        return go.Figure()
    C_PINK  = "#EC4899"
    C_ACCENT= "#6366F1"
    C_RED   = "#EF4444"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
                        subplot_titles=["Temperatura del horno (°C)", "Ocupación del horno"])
    fig.add_trace(go.Scatter(
        x=df_s["tiempo"], y=df_s["temperatura"],
        mode="lines", name="Temperatura",
        line=dict(color=C_PINK, width=1.8),
        fill="tozeroy", fillcolor="rgba(236,72,153,0.07)",
        hovertemplate="t=%{x:.0f} min<br><b>%{y:.1f}°C</b><extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=200, line_dash="dash", line_color=C_RED,
                  annotation_text="Límite 200°C", annotation_font_color=C_RED, row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_s["tiempo"], y=df_s["horno_ocup"],
        mode="lines", name="Estaciones ocupadas",
        fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
        line=dict(color=C_ACCENT, width=1.8),
    ), row=2, col=1)
    fig.update_layout(
        **theme, height=460,
        title=dict(text="Sensores virtuales — monitor del horno en tiempo real", x=0.5),
    )
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="Estaciones", row=2, col=1)
    fig.update_xaxes(title_text="Tiempo simulado (min)", row=2, col=1)
    return fig
