"""
demanda.py
==========
Funciones de análisis y visualización de demanda histórica.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datos import PRODUCTOS, MESES, MESES_C, DEM_HISTORICA, PROD_COLORS


def get_resumen_demanda():
    """Retorna un DataFrame con la demanda histórica."""
    return pd.DataFrame(DEM_HISTORICA, index=MESES_C).reset_index().rename(columns={"index": "Mes"})


def get_kpis_demanda():
    """Calcula KPIs básicos de demanda."""
    total = sum(sum(v) for v in DEM_HISTORICA.values())
    pi    = max(range(12), key=lambda i: sum(DEM_HISTORICA[p][i] for p in PRODUCTOS))
    hh_anual = sum(
        DEM_HISTORICA[p][i] * 0.866 if p == "Brownies" else
        DEM_HISTORICA[p][i] * 0.175 if p in ("Mantecadas","Mantecadas_Amapola","Torta_Naranja") else
        DEM_HISTORICA[p][i] * 0.312
        for p in PRODUCTOS for i in range(12)
    )
    return {
        "total_anual":  total,
        "n_productos":  len(PRODUCTOS),
        "mes_pico":     MESES_C[pi],
        "avg_mensual":  total // 12,
        "hh_anuales":   round(hh_anual),
    }


def fig_barras_demanda(theme: dict) -> go.Figure:
    """Gráfico de barras agrupadas por producto y mes."""
    fig = go.Figure()
    for p in PRODUCTOS:
        fig.add_trace(go.Bar(
            x=MESES_C, y=DEM_HISTORICA[p],
            name=p.replace("_", " "),
            marker_color=PROD_COLORS[p],
            marker_line_width=0,
            opacity=0.88,
            hovertemplate=f"<b>{p.replace('_',' ')}</b><br>%{{x}}<br><b>%{{y:,}}</b> und<extra></extra>",
        ))
    fig.update_layout(
        **theme,
        barmode="group",
        height=400,
        title=dict(text="Demanda histórica por producto", x=0.5),
        xaxis_title="",
        yaxis_title="Unidades",
        bargap=0.2,
        bargroupgap=0.05,
    )
    return fig


def fig_heatmap_demanda(theme: dict) -> go.Figure:
    """Mapa de calor de estacionalidad."""
    z = [[DEM_HISTORICA[p][i] for i in range(12)] for p in PRODUCTOS]
    colorscale = [
        [0.0, "#EEF2FF"], [0.3, "#A5B4FC"],
        [0.65, "#6366F1"], [1.0, "#3730A3"],
    ]
    fig = go.Figure(go.Heatmap(
        z=z, x=MESES_C, y=[p.replace("_", " ") for p in PRODUCTOS],
        colorscale=colorscale, showscale=True,
        hovertemplate="%{y}<br>%{x}<br><b>%{z:,}</b> und<extra></extra>",
        colorbar=dict(outlinewidth=0, thickness=12),
    ))
    fig.update_layout(
        **theme,
        height=320,
        title=dict(text="Mapa de calor — estacionalidad", x=0.5),
        margin=dict(l=120, r=60, t=50, b=40),
    )
    return fig
