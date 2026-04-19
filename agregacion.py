"""
agregacion.py
=============
Planeación agregada por horas-hombre usando PuLP / CBC.
"""

import pandas as pd
import plotly.graph_objects as go
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, PULP_CBC_CMD
from datos import PRODUCTOS, MESES, MESES_C, DEM_HISTORICA, HORAS_PRODUCTO, PARAMS_AGRE


# ── Motor de optimización ─────────────────────────────────────────────────────

def calcular_dem_horas(factor: float = 1.0) -> dict:
    """Convierte demanda de unidades a horas-hombre por mes."""
    return {
        mes: round(
            sum(DEM_HISTORICA[p][i] * HORAS_PRODUCTO[p] for p in PRODUCTOS) * factor, 4
        )
        for i, mes in enumerate(MESES)
    }


def run_agregacion(dem_horas: dict, params: dict = None):
    """
    Resuelve el modelo de planeación agregada.

    Retorna
    -------
    df_agr : pd.DataFrame
        Plan mensual en H-H.
    costo : float
        Valor de la función objetivo (COP).
    """
    if params is None:
        params = PARAMS_AGRE.copy()

    Ct, Ht, PIt = params["Ct"], params["Ht"], params["PIt"]
    CRt, COt    = params["CRt"], params["COt"]
    Wm, Wd      = params["CW_mas"], params["CW_menos"]
    M, LRi      = params["M"], params["LR_inicial"]

    mdl = LpProblem("Agregacion", LpMinimize)
    P      = LpVariable.dicts("P",  MESES, lowBound=0)
    I      = LpVariable.dicts("I",  MESES, lowBound=0)
    S      = LpVariable.dicts("S",  MESES, lowBound=0)
    LR     = LpVariable.dicts("LR", MESES, lowBound=0)
    LO     = LpVariable.dicts("LO", MESES, lowBound=0)
    LU     = LpVariable.dicts("LU", MESES, lowBound=0)
    NI     = LpVariable.dicts("NI", MESES)
    Wmas   = LpVariable.dicts("Wm", MESES, lowBound=0)
    Wmenos = LpVariable.dicts("Wd", MESES, lowBound=0)

    mdl += lpSum(
        Ct*P[t] + Ht*I[t] + PIt*S[t] + CRt*LR[t] + COt*LO[t] + Wm*Wmas[t] + Wd*Wmenos[t]
        for t in MESES
    )

    for idx, t in enumerate(MESES):
        d  = dem_horas[t]
        tp = MESES[idx - 1] if idx > 0 else None
        if idx == 0:
            mdl += NI[t] == P[t] - d
        else:
            mdl += NI[t] == NI[tp] + P[t] - d
        mdl += NI[t] == I[t] - S[t]
        mdl += LU[t] + LO[t] == M * P[t]
        mdl += LU[t] <= LR[t]
        if idx == 0:
            mdl += LR[t] == LRi + Wmas[t] - Wmenos[t]
        else:
            mdl += LR[t] == LR[tp] + Wmas[t] - Wmenos[t]

    mdl.solve(PULP_CBC_CMD(msg=False))
    costo = value(mdl.objective)

    fin_l = []
    for idx, t in enumerate(MESES):
        ini = 0.0 if idx == 0 else fin_l[-1]
        fin_l.append(ini + P[t].varValue - dem_horas[t])

    df = pd.DataFrame({
        "Mes":                MESES,
        "Demanda_HH":         [round(dem_horas[t], 2) for t in MESES],
        "Produccion_HH":      [round(P[t].varValue, 2) for t in MESES],
        "Backlog_HH":         [round(S[t].varValue, 2) for t in MESES],
        "Horas_Regulares":    [round(LR[t].varValue, 2) for t in MESES],
        "Horas_Extras":       [round(LO[t].varValue, 2) for t in MESES],
        "Inventario_Final_HH":[round(v, 2) for v in fin_l],
        "Contratacion":       [round(Wmas[t].varValue, 2) for t in MESES],
        "Despidos":           [round(Wmenos[t].varValue, 2) for t in MESES],
    })
    return df, costo


# ── Visualización ─────────────────────────────────────────────────────────────

def fig_plan_agregado(df_agr: pd.DataFrame, costo: float, theme: dict) -> go.Figure:
    """Gráfico de barras apiladas + líneas para el plan agregado."""
    C_GREEN = "#10B981"
    C_AMBER = "#F59E0B"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=MESES_C, y=df_agr["Inventario_Final_HH"],
        name="Inventario H-H", marker_color="#A5B4FC",
        marker_line_width=0, opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        x=MESES_C, y=df_agr["Produccion_HH"],
        name="Producción H-H", marker_color="#6366F1",
        marker_line_width=0, opacity=0.88,
    ))
    fig.add_trace(go.Scatter(
        x=MESES_C, y=df_agr["Demanda_HH"], mode="lines+markers",
        name="Demanda H-H",
        line=dict(color=C_GREEN, dash="dash", width=2.5),
        marker=dict(size=7, color=C_GREEN, line=dict(color="white", width=1.5)),
    ))
    fig.add_trace(go.Scatter(
        x=MESES_C, y=df_agr["Horas_Regulares"], mode="lines",
        name="Cap. regular",
        line=dict(color=C_AMBER, dash="dot", width=2),
    ))
    fig.update_layout(
        **theme,
        barmode="stack",
        height=420,
        title=dict(text=f"Plan agregado — costo óptimo COP ${costo:,.0f}", x=0.5),
        yaxis_title="Horas-Hombre (H-H)",
    )
    return fig
