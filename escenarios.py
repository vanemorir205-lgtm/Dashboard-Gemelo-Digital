"""
escenarios.py
=============
Análisis de escenarios what-if: definición, ejecución y comparación.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datos import CAPACIDAD_BASE, TAMANO_LOTE_BASE
from simulacion import run_simulacion, calc_kpis, calc_utilizacion


# ── Definición de escenarios ──────────────────────────────────────────────────

ESCENARIOS = {
    "base":        {"label": "Base",                          "fd": 1.0, "falla": False, "ft": 1.0,  "dh": 0,  "fl": 1.0},
    "demanda_20":  {"label": "Demanda +20%",                  "fd": 1.2, "falla": False, "ft": 1.0,  "dh": 0,  "fl": 1.0},
    "falla_horno": {"label": "Falla en horno",                "fd": 1.0, "falla": True,  "ft": 1.0,  "dh": 0,  "fl": 1.0},
    "red_cap":     {"label": "Reducir cap. horno",            "fd": 1.0, "falla": False, "ft": 1.0,  "dh": -1, "fl": 1.0},
    "doble_turno": {"label": "Doble turno (−20%)",            "fd": 1.0, "falla": False, "ft": 0.80, "dh": 0,  "fl": 1.0},
    "lote_grande": {"label": "Lotes +50%",                    "fd": 1.0, "falla": False, "ft": 1.0,  "dh": 0,  "fl": 1.5},
    "optimizado":  {"label": "Optimizado (+1 horno, −15%)",   "fd": 1.0, "falla": False, "ft": 0.85, "dh": 1,  "fl": 1.0},
}

ESC_OPTIONS = [{"label": v["label"], "value": k} for k, v in ESCENARIOS.items()]


# ── Motor de escenarios ───────────────────────────────────────────────────────

def correr_escenario(nombre: str, plan_mes: dict) -> dict:
    """
    Ejecuta un escenario y retorna sus KPIs y utilización serializados.

    Retorna
    -------
    {"kpis": str_json, "util": str_json}
    """
    cfg     = ESCENARIOS.get(nombre, ESCENARIOS["base"])
    plan_aj = {p: max(int(u * cfg["fd"]), 0) for p, u in plan_mes.items()}
    cap_r   = {**CAPACIDAD_BASE, "horno": max(CAPACIDAD_BASE["horno"] + cfg["dh"], 1)}
    tam_l   = {p: max(int(t * cfg["fl"]), 1) for p, t in TAMANO_LOTE_BASE.items()}
    df_l, df_u, _ = run_simulacion(plan_aj, cap_r, cfg["falla"], cfg["ft"], tam_l)
    dk = calc_kpis(df_l, plan_aj)
    du = calc_utilizacion(df_u)
    return {
        "kpis": dk.to_json() if not dk.empty else "{}",
        "util": du.to_json() if not du.empty else "{}",
    }


def correr_escenarios_seleccionados(nombres: list, plan_mes: dict, acumulado: dict) -> dict:
    """Corre varios escenarios y acumula resultados."""
    resultado = dict(acumulado or {})
    for nm in nombres:
        resultado[nm] = correr_escenario(nm, plan_mes)
    return resultado


# ── Visualización ─────────────────────────────────────────────────────────────

def fig_comparacion_escenarios(esc_store: dict, theme: dict) -> go.Figure:
    """
    Compara escenarios en 4 métricas: throughput, lead time,
    cumplimiento y utilización máxima.
    """
    if not esc_store:
        return go.Figure()

    filas = []
    for nm, v in esc_store.items():
        dk = pd.read_json(v["kpis"]) if v.get("kpis", "{}") != "{}" else pd.DataFrame()
        du = pd.read_json(v["util"]) if v.get("util", "{}") != "{}" else pd.DataFrame()
        if dk.empty:
            continue
        fila = {"Escenario": ESCENARIOS.get(nm, {}).get("label", nm)}
        for col in ["Throughput (und/h)", "Lead Time (min/lote)", "WIP Prom", "Cumplimiento %"]:
            if col in dk.columns:
                fila[col] = round(dk[col].mean(), 2)
        if not du.empty and "Utilización_%" in du.columns:
            fila["Util Máx %"] = round(du["Utilización_%"].max(), 2)
        filas.append(fila)

    if not filas:
        return go.Figure()

    df = pd.DataFrame(filas)
    metricas = [
        ("Throughput (und/h)",   "Throughput (und/h)"),
        ("Lead Time (min/lote)", "Lead Time (min)"),
        ("Cumplimiento %",       "Cumplimiento (%)"),
        ("Util Máx %",           "Util. máx (%)"),
    ]
    paleta = ["#6366F1", "#0EA5E9", "#10B981", "#F59E0B", "#EC4899", "#A78BFA", "#34D399"]
    fig    = make_subplots(rows=2, cols=2,
                           subplot_titles=[m[1] for m in metricas],
                           vertical_spacing=0.18)
    for i, (col, _) in enumerate(metricas):
        r, c = i // 2 + 1, i % 2 + 1
        if col not in df.columns:
            continue
        fig.add_trace(go.Bar(
            x=df["Escenario"], y=df[col],
            marker_color=[paleta[j % len(paleta)] for j in range(len(df))],
            marker_line_width=0, opacity=0.88,
            text=df[col].apply(lambda v: f"{v:.1f}"),
            textposition="outside",
            showlegend=False,
        ), row=r, col=c)

    fig.update_layout(
        **theme, height=500,
        title=dict(text="Comparación de escenarios what-if", x=0.5),
    )
    fig.update_xaxes(tickangle=20, tickfont=dict(size=9))
    return fig
