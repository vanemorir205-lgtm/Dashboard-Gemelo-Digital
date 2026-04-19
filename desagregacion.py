"""
desagregacion.py
================
Desagregación del plan agregado por producto usando PuLP.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
from datos import PRODUCTOS, MESES, MESES_C, DEM_HISTORICA, HORAS_PRODUCTO, INV_INICIAL, PROD_COLORS


# ── Motor de optimización ─────────────────────────────────────────────────────

def run_desagregacion(prod_hh: dict, factor_demanda: float = 1.0) -> dict:
    """
    Desagrega las horas-hombre del plan agregado en unidades por producto.

    Parámetros
    ----------
    prod_hh : dict
        {mes: horas_produccion} del plan agregado.
    factor_demanda : float
        Escala la demanda histórica.

    Retorna
    -------
    dict {producto: pd.DataFrame}
    """
    mdl = LpProblem("Desagregacion", LpMinimize)
    X = {(p, t): LpVariable(f"X_{p}_{t}", lowBound=0) for p in PRODUCTOS for t in MESES}
    I = {(p, t): LpVariable(f"I_{p}_{t}", lowBound=0) for p in PRODUCTOS for t in MESES}
    S = {(p, t): LpVariable(f"S_{p}_{t}", lowBound=0) for p in PRODUCTOS for t in MESES}

    mdl += lpSum(100000 * I[p, t] + 150000 * S[p, t] for p in PRODUCTOS for t in MESES)

    for idx, t in enumerate(MESES):
        tp = MESES[idx - 1] if idx > 0 else None
        mdl += (lpSum(HORAS_PRODUCTO[p] * X[p, t] for p in PRODUCTOS) <= prod_hh[t])
        for p in PRODUCTOS:
            d = int(DEM_HISTORICA[p][idx] * factor_demanda)
            if idx == 0:
                mdl += I[p, t] - S[p, t] == INV_INICIAL[p] + X[p, t] - d
            else:
                mdl += I[p, t] - S[p, t] == I[p, tp] - S[p, tp] + X[p, t] - d

    mdl.solve(PULP_CBC_CMD(msg=False))

    out = {}
    for p in PRODUCTOS:
        rows = []
        for idx, t in enumerate(MESES):
            xv  = round(X[p, t].varValue or 0, 2)
            iv  = round(I[p, t].varValue or 0, 2)
            sv  = round(S[p, t].varValue or 0, 2)
            ini = INV_INICIAL[p] if idx == 0 else round(I[p, MESES[idx - 1]].varValue or 0, 2)
            rows.append({
                "Mes":      t,
                "Demanda":  int(DEM_HISTORICA[p][idx] * factor_demanda),
                "Produccion": xv,
                "Inv_Ini":  ini,
                "Inv_Fin":  iv,
                "Backlog":  sv,
            })
        out[p] = pd.DataFrame(rows)
    return out


# ── Visualización ─────────────────────────────────────────────────────────────

def fig_desagregacion(desag_dict: dict, mes_sel: str, theme: dict) -> go.Figure:
    """Subplots 3×2 con producción vs demanda por producto."""
    C_TEXT3 = "#9CA3AF"
    C_AMBER = "#F59E0B"

    fig = make_subplots(
        rows=3, cols=2,
        vertical_spacing=0.09, horizontal_spacing=0.10,
        subplot_titles=[p.replace("_", " ") for p in PRODUCTOS],
    )
    for idx, p in enumerate(PRODUCTOS):
        r, c = idx // 2 + 1, idx % 2 + 1
        df   = desag_dict[p]
        col  = PROD_COLORS[p]

        fig.add_trace(go.Bar(
            x=MESES_C, y=df["Produccion"],
            marker_color=col, marker_line_width=0, opacity=0.85,
            showlegend=False,
            hovertemplate="%{x}<br><b>%{y:,.0f}</b> und<extra></extra>",
        ), row=r, col=c)

        fig.add_trace(go.Scatter(
            x=MESES_C, y=df["Demanda"], mode="lines+markers",
            line=dict(color=C_TEXT3, dash="dash", width=1.5),
            marker=dict(size=4), showlegend=False,
        ), row=r, col=c)

        if mes_sel in MESES:
            mi      = MESES.index(mes_sel)
            mes_row = df[df["Mes"] == mes_sel]
            if not mes_row.empty:
                fig.add_trace(go.Scatter(
                    x=[MESES_C[mi]], y=[mes_row["Produccion"].values[0]],
                    mode="markers",
                    marker=dict(size=14, color=C_AMBER, symbol="star",
                                line=dict(color="white", width=1.5)),
                    showlegend=False,
                ), row=r, col=c)

    fig.update_layout(
        **theme, height=700, barmode="group",
        title=dict(text="Desagregación por producto — unidades/mes", x=0.5),
    )
    for i in range(1, 4):
        for j in range(1, 3):
            fig.update_xaxes(tickfont=dict(size=8), row=i, col=j)
            fig.update_yaxes(tickfont=dict(size=8), title_text="und", row=i, col=j)
    return fig
