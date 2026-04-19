"""
app.py
======
Dashboard Gemelo Digital — Dora del Hoyo
=========================================
Interfaz principal que integra demanda, agregación, desagregación,
simulación, KPIs, sensores y análisis de escenarios.
 
INSTALACIÓN:
    pip install dash dash-bootstrap-components simpy pulp pandas numpy plotly statsmodels
 
EJECUCIÓN:
    python app.py
    → Abrir http://127.0.0.1:8050
"""
 
import warnings
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
 
warnings.filterwarnings("ignore")
 
# ── Imports de módulos propios ────────────────────────────────────────────────
from datos import (
    PRODUCTOS, MESES, MESES_C, DEM_HISTORICA, PROD_COLORS,
    CAPACIDAD_BASE, TAMANO_LOTE_BASE, PARAMS_AGRE,
)
from demanda import get_kpis_demanda, get_resumen_demanda, fig_barras_demanda, fig_heatmap_demanda
from agregacion import calcular_dem_horas, run_agregacion, fig_plan_agregado
from desagregacion import run_desagregacion, fig_desagregacion
from simulacion import (
    run_simulacion, calc_utilizacion, calc_kpis,
    fig_gantt, fig_colas, fig_utilizacion, fig_kpis_radar, fig_sensores,
)
from escenarios import (
    ESCENARIOS, ESC_OPTIONS, correr_escenarios_seleccionados,
    fig_comparacion_escenarios,
)
 
# ─────────────────────────────────────────────────────────────────────────────
# TEMA VISUAL GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
 
THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'DM Mono', monospace", color="#E2E8F0", size=11),
    xaxis=dict(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.2)",
               tickfont=dict(color="#94A3B8")),
    yaxis=dict(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.2)",
               tickfont=dict(color="#94A3B8")),
    legend=dict(bgcolor="rgba(15,23,42,0.6)", font=dict(size=10, color="#CBD5E1"),
                bordercolor="rgba(148,163,184,0.15)", borderwidth=1),
    margin=dict(l=52, r=24, t=58, b=48),
    colorway=["#6366F1", "#0EA5E9", "#10B981", "#F59E0B", "#EC4899", "#A78BFA"],
    title=dict(font=dict(family="'Outfit', sans-serif", size=14, color="#F1F5F9")),
    hoverlabel=dict(bgcolor="#1E293B", font=dict(color="#F1F5F9", size=11)),
)
 
# ─────────────────────────────────────────────────────────────────────────────
# ESTILOS BASE
# ─────────────────────────────────────────────────────────────────────────────
 
CARD = {
    "background": "rgba(30,41,59,0.7)",
    "border": "1px solid rgba(148,163,184,0.12)",
    "borderRadius": "12px",
    "padding": "20px 24px",
    "backdropFilter": "blur(12px)",
}
 
CARD_SM = {**CARD, "padding": "14px 18px"}
 
LABEL = {
    "color": "#64748B",
    "fontSize": "9px",
    "fontFamily": "'DM Mono', monospace",
    "letterSpacing": "0.14em",
    "textTransform": "uppercase",
    "marginBottom": "6px",
    "display": "block",
}
 
INPUT_S = {
    "background": "rgba(15,23,42,0.8)",
    "color": "#E2E8F0",
    "border": "1px solid rgba(148,163,184,0.2)",
    "borderRadius": "8px",
    "fontFamily": "'DM Mono', monospace",
    "fontSize": "12px",
}
 
# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTES REUTILIZABLES
# ─────────────────────────────────────────────────────────────────────────────
 
def kpi_card(label, value, unit="", color="#6366F1", icon="▸"):
    return html.Div([
        html.Div(icon + "  " + label, style={**LABEL}),
        html.Div([
            html.Span(str(value), style={
                "fontSize": "28px", "fontWeight": "700", "color": color,
                "fontFamily": "'Outfit', sans-serif", "lineHeight": "1",
            }),
            html.Span("  " + unit, style={
                "fontSize": "11px", "color": "#64748B", "marginLeft": "4px",
                "fontFamily": "'DM Mono', monospace",
            }) if unit else None,
        ], style={"display": "flex", "alignItems": "baseline", "marginTop": "2px"}),
    ], style={
        **CARD_SM,
        "minWidth": "130px",
        "borderTop": f"2px solid {color}",
        "transition": "transform 0.2s",
    })
 
def page_header(title, subtitle=""):
    return html.Div([
        html.Div(subtitle, style={
            "fontSize": "9px", "letterSpacing": "0.22em", "color": "#6366F1",
            "fontFamily": "'DM Mono', monospace", "textTransform": "uppercase",
            "marginBottom": "4px",
        }),
        html.H2(title, style={
            "fontFamily": "'Outfit', sans-serif", "fontWeight": "800",
            "fontSize": "26px", "color": "#F1F5F9", "margin": "0 0 2px 0",
            "letterSpacing": "-0.01em",
        }),
        html.Div(style={
            "height": "3px", "width": "48px",
            "background": "linear-gradient(90deg, #6366F1, #0EA5E9)",
            "borderRadius": "2px", "marginTop": "8px",
        }),
    ], style={"marginBottom": "24px"})
 
def section_label(text):
    return html.Div([
        html.Span(text, style={
            "fontSize": "9px", "letterSpacing": "0.18em", "color": "#6366F1",
            "fontFamily": "'DM Mono', monospace", "textTransform": "uppercase",
        }),
        html.Div(style={
            "height": "1px", "flex": "1", "marginLeft": "12px",
            "background": "rgba(99,102,241,0.2)",
        }),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px", "marginTop": "28px"})
 
def tabla_dash(df, id_tabla, page_size=12):
    if df is None or (hasattr(df, "empty") and df.empty):
        return html.Div("Sin datos", style={"color": "#475569", "fontFamily": "'DM Mono', monospace"})
    df_r = df.copy()
    for col in df_r.select_dtypes(include="float").columns:
        df_r[col] = df_r[col].round(3)
    return dash_table.DataTable(
        id=id_tabla,
        columns=[{"name": c, "id": c} for c in df_r.columns],
        data=df_r.to_dict("records"),
        page_size=page_size,
        style_table={"overflowX": "auto", "borderRadius": "8px", "overflow": "hidden"},
        style_header={
            "backgroundColor": "#0F172A", "color": "#6366F1",
            "fontFamily": "'DM Mono', monospace", "fontSize": "9px",
            "border": "1px solid rgba(99,102,241,0.2)", "letterSpacing": "0.1em",
            "textTransform": "uppercase", "padding": "10px 12px",
        },
        style_cell={
            "backgroundColor": "rgba(15,23,42,0.8)", "color": "#CBD5E1",
            "fontFamily": "'DM Mono', monospace", "fontSize": "11px",
            "border": "1px solid rgba(148,163,184,0.08)",
            "padding": "8px 12px", "textAlign": "right",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgba(30,41,59,0.4)"},
            {"if": {"state": "selected"}, "backgroundColor": "rgba(99,102,241,0.15)",
             "border": "1px solid rgba(99,102,241,0.4)"},
        ],
        style_filter={"backgroundColor": "#0F172A", "color": "#E2E8F0"},
        filter_action="native",
        sort_action="native",
    )
 
def no_data_msg():
    return html.Div([
        html.Div("◈", style={"fontSize": "56px", "color": "rgba(99,102,241,0.2)",
                              "textAlign": "center", "padding": "48px 0 8px",
                              "fontFamily": "'Outfit', sans-serif"}),
        html.Div("Ejecuta el pipeline primero", style={
            "textAlign": "center", "color": "#475569",
            "fontSize": "14px", "fontFamily": "'Outfit', sans-serif",
        }),
        html.Div("Configura los parámetros y presiona  ▶  EJECUTAR", style={
            "textAlign": "center", "color": "#334155", "fontSize": "11px",
            "fontFamily": "'DM Mono', monospace", "marginTop": "6px",
        }),
    ])
 
# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
 
NAV_ITEMS = [
    ("01", "Demanda",       "tab-demanda",     "📊"),
    ("02", "Planeación",    "tab-agregacion",  "📋"),
    ("03", "Desagregación", "tab-desag",       "🔀"),
    ("04", "Simulación",    "tab-sim",         "⚙️"),
    ("05", "KPIs",          "tab-kpis",        "📈"),
    ("06", "Sensores",      "tab-sensores",    "🌡️"),
    ("07", "Escenarios",    "tab-escenarios",  "🔬"),
]
 
sidebar = html.Div([
    # Logo
    html.Div([
        html.Div("◈", style={
            "fontSize": "32px", "color": "#6366F1",
            "fontFamily": "'Outfit', sans-serif", "lineHeight": "1",
        }),
        html.Div("DORA", style={
            "fontSize": "18px", "fontWeight": "800", "color": "#F1F5F9",
            "fontFamily": "'Outfit', sans-serif", "letterSpacing": "0.08em",
            "lineHeight": "1", "marginTop": "4px",
        }),
        html.Div("DEL HOYO", style={
            "fontSize": "18px", "fontWeight": "800", "color": "#F1F5F9",
            "fontFamily": "'Outfit', sans-serif", "letterSpacing": "0.08em",
            "lineHeight": "1",
        }),
        html.Div("GEMELO DIGITAL", style={
            "fontSize": "8px", "color": "#6366F1", "letterSpacing": "0.2em",
            "fontFamily": "'DM Mono', monospace", "marginTop": "6px",
        }),
    ], style={
        "padding": "28px 20px 22px",
        "borderBottom": "1px solid rgba(148,163,184,0.1)",
        "marginBottom": "8px",
    }),
 
    # Nav links
    html.Div([
        html.Button(
            [
                html.Span(icon + " ", style={"fontSize": "14px", "marginRight": "4px"}),
                html.Div([
                    html.Span(num, style={
                        "fontSize": "8px", "color": "#475569",
                        "fontFamily": "'DM Mono', monospace",
                        "display": "block", "lineHeight": "1",
                    }),
                    html.Span(label, style={"fontSize": "12px", "lineHeight": "1.2"}),
                ]),
            ],
            id=f"btn-{tab}",
            n_clicks=0,
            style={
                "background": "transparent", "border": "none", "color": "#94A3B8",
                "width": "100%", "textAlign": "left",
                "padding": "11px 20px", "cursor": "pointer",
                "fontFamily": "'Outfit', sans-serif", "fontWeight": "600",
                "display": "flex", "alignItems": "center", "gap": "8px",
                "transition": "all 0.15s", "borderRadius": "0",
                "borderLeft": "3px solid transparent",
            },
        )
        for num, label, tab, icon in NAV_ITEMS
    ], id="nav-links"),
 
    # Footer
    html.Div([
        html.Div("v2.0 · 2025", style={
            "fontSize": "9px", "color": "#334155",
            "fontFamily": "'DM Mono', monospace",
        }),
        html.Div("Planeación & Sim.", style={
            "fontSize": "9px", "color": "#334155",
            "fontFamily": "'DM Mono', monospace",
        }),
    ], style={
        "position": "absolute", "bottom": "0", "width": "100%",
        "padding": "14px 20px", "borderTop": "1px solid rgba(148,163,184,0.08)",
    }),
], style={
    "width": "200px", "minHeight": "100vh",
    "background": "linear-gradient(180deg, #0F172A 0%, #0B1120 100%)",
    "borderRight": "1px solid rgba(148,163,184,0.08)",
    "display": "flex", "flexDirection": "column",
    "position": "fixed", "top": "0", "left": "0", "zIndex": "200",
})
 
# ─────────────────────────────────────────────────────────────────────────────
# PANEL DE PARÁMETROS (solo visible en tabs que lo necesitan)
# ─────────────────────────────────────────────────────────────────────────────
 
param_panel = html.Div(id="param-panel", children=[
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Span("MES A SIMULAR", style=LABEL),
                dcc.Dropdown(
                    id="dd-mes",
                    options=[{"label": m, "value": i} for i, m in enumerate(MESES)],
                    value=1, clearable=False,
                    style={**INPUT_S, "minWidth": "155px"},
                    className="dd-dark",
                ),
            ], width="auto"),
            dbc.Col([
                html.Span("FACTOR DEMANDA", style=LABEL),
                html.Div([
                    dcc.Slider(
                        id="sl-demanda", min=0.5, max=2.0, step=0.1, value=1.0,
                        marks={0.5: {"label": "0.5×", "style": {"color": "#64748B", "fontSize": "10px"}},
                               1.0: {"label": "1×",   "style": {"color": "#6366F1", "fontSize": "10px"}},
                               1.5: {"label": "1.5×", "style": {"color": "#64748B", "fontSize": "10px"}},
                               2.0: {"label": "2×",   "style": {"color": "#64748B", "fontSize": "10px"}}},
                        tooltip={"placement": "top", "always_visible": True},
                        className="slider-indigo",
                    ),
                ], style={"paddingTop": "4px", "minWidth": "200px"}),
            ], width=3),
            dbc.Col([
                html.Span("CAPACIDAD HORNO", style=LABEL),
                html.Div([
                    dcc.Slider(
                        id="sl-horno", min=1, max=6, step=1, value=3,
                        marks={i: {"label": str(i), "style": {"color": "#64748B", "fontSize": "10px"}}
                               for i in range(1, 7)},
                        tooltip={"placement": "top", "always_visible": True},
                        className="slider-indigo",
                    ),
                ], style={"paddingTop": "4px", "minWidth": "160px"}),
            ], width=3),
            dbc.Col([
                html.Span("OPCIONES", style=LABEL),
                dbc.Checklist(
                    id="chk-opciones",
                    options=[
                        {"label": " Falla en horno",       "value": "falla"},
                        {"label": " Doble turno (−20%)",   "value": "turno"},
                    ],
                    value=[], switch=True,
                    style={"color": "#CBD5E1", "fontSize": "11px",
                           "fontFamily": "'DM Mono', monospace"},
                ),
            ], width="auto"),
            dbc.Col([
                html.Div(style={"height": "20px"}),
                html.Button(
                    "▶  EJECUTAR",
                    id="btn-run", n_clicks=0,
                    style={
                        "background": "linear-gradient(135deg, #6366F1, #0EA5E9)",
                        "color": "#fff", "border": "none",
                        "padding": "9px 22px",
                        "fontFamily": "'Outfit', sans-serif",
                        "fontWeight": "700", "fontSize": "12px",
                        "letterSpacing": "0.1em", "borderRadius": "8px",
                        "cursor": "pointer", "whiteSpace": "nowrap",
                        "boxShadow": "0 4px 14px rgba(99,102,241,0.35)",
                    },
                ),
            ], width="auto", style={"display": "flex", "alignItems": "flex-end"}),
        ], className="g-3", align="end"),
        html.Div(id="run-status", style={
            "marginTop": "10px", "fontSize": "11px", "color": "#10B981",
            "fontFamily": "'DM Mono', monospace",
        }),
    ], style={**CARD, "marginBottom": "0"}),
])
 
# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
 
EXTERNAL = [
    "https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap",
    dbc.themes.BOOTSTRAP,
]
 
app = dash.Dash(
    __name__,
    external_stylesheets=EXTERNAL,
    suppress_callback_exceptions=True,
    title="Gemelo Digital — Dora del Hoyo",
)
server = app.server  # para Render/Gunicorn
 
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        *, *::before, *::after { box-sizing: border-box; }
        html, body { margin: 0; padding: 0; background: #080E1C; color: #E2E8F0; }
 
        /* Scrollbar */
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: #0F172A; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
 
        /* Dropdown oscuro */
        .dd-dark .Select-control { background: rgba(15,23,42,0.9) !important; border-color: rgba(148,163,184,0.2) !important; }
        .dd-dark .Select-menu-outer { background: #1E293B !important; border-color: rgba(148,163,184,0.15) !important; }
        .dd-dark .Select-option { background: #1E293B !important; color: #CBD5E1 !important; }
        .dd-dark .Select-option.is-focused { background: rgba(99,102,241,0.18) !important; }
        .dd-dark .Select-value-label { color: #E2E8F0 !important; }
        .dd-dark .Select-arrow { border-top-color: #64748B !important; }
        .Select-placeholder { color: #64748B !important; }
 
        /* Sliders */
        .slider-indigo .rc-slider-track { background: linear-gradient(90deg, #6366F1, #0EA5E9) !important; }
        .slider-indigo .rc-slider-handle { border-color: #6366F1 !important; background: #6366F1 !important; box-shadow: 0 0 0 4px rgba(99,102,241,0.2) !important; }
        .rc-slider-tooltip-inner { background: #1E293B !important; color: #E2E8F0 !important; border: 1px solid rgba(99,102,241,0.3) !important; font-family: 'DM Mono', monospace !important; font-size: 10px !important; }
        .rc-slider-rail { background: rgba(148,163,184,0.12) !important; }
 
        /* Checklist switch */
        .form-check-input:checked { background-color: #6366F1 !important; border-color: #6366F1 !important; }
        .form-check-input { background-color: rgba(148,163,184,0.15) !important; border-color: rgba(148,163,184,0.3) !important; }
 
        /* Nav buttons active */
        .nav-active { color: #A5B4FC !important; background: rgba(99,102,241,0.1) !important; border-left: 3px solid #6366F1 !important; }
 
        /* Loading dot */
        ._dash-loading { color: #6366F1 !important; }
 
        /* Plotly modebar */
        .modebar { background: transparent !important; }
        .modebar-btn path { fill: #475569 !important; }
 
        /* Noise overlay */
        .main-content::before {
            content: '';
            position: fixed; top: 0; left: 200px; right: 0; bottom: 0;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.018'/%3E%3C/svg%3E");
            pointer-events: none; z-index: 0; opacity: 0.4;
        }
 
        /* Animated gradient bg */
        body {
            background: radial-gradient(ellipse at 20% 50%, rgba(99,102,241,0.05) 0%, transparent 50%),
                        radial-gradient(ellipse at 80% 20%, rgba(14,165,233,0.04) 0%, transparent 50%),
                        #080E1C;
        }
 
        /* Dash table */
        .dash-spreadsheet-container .dash-spreadsheet-inner td { background: rgba(15,23,42,0.8) !important; }
        .previous-page, .next-page, .last-page, .first-page { color: #6366F1 !important; }
        .page-number { color: #94A3B8 !important; }
 
        /* Checklist labels */
        .form-check-label { color: #CBD5E1 !important; font-family: 'DM Mono', monospace; font-size: 11px; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''
 
# ── Stores ─────────────────────────────────────────────────────────────────
 
stores = html.Div([
    dcc.Store(id="store-active-tab", data="tab-demanda"),
    dcc.Store(id="store-agr",        data=None),
    dcc.Store(id="store-desag",      data=None),
    dcc.Store(id="store-sim",        data=None),
    dcc.Store(id="store-util",       data=None),
    dcc.Store(id="store-kpis",       data=None),
    dcc.Store(id="store-sensores",   data=None),
    dcc.Store(id="store-plan-mes",   data=None),
    dcc.Store(id="store-esc",        data={}),
    dcc.Interval(id="auto-run", interval=800, max_intervals=1),
])
 
# ── Layout ─────────────────────────────────────────────────────────────────
 
app.layout = html.Div([
    stores,
    sidebar,
    html.Div([
        # Header superior
        html.Div([
            html.Div(id="header-breadcrumb", style={
                "fontSize": "9px", "letterSpacing": "0.2em", "color": "#334155",
                "fontFamily": "'DM Mono', monospace", "marginBottom": "2px",
            }),
            html.Div(id="header-title", children="DEMANDA", style={
                "fontFamily": "'Outfit', sans-serif", "fontWeight": "800",
                "fontSize": "13px", "color": "#64748B", "letterSpacing": "0.12em",
            }),
        ], style={
            "padding": "14px 32px 12px",
            "borderBottom": "1px solid rgba(148,163,184,0.07)",
            "background": "rgba(15,23,42,0.6)",
            "display": "flex", "alignItems": "center", "gap": "24px",
            "backdropFilter": "blur(8px)",
            "position": "sticky", "top": "0", "zIndex": "100",
        }),
 
        # Zona de contenido
        html.Div([
            # Panel de parámetros (condicional)
            html.Div(id="params-wrapper"),
 
            # Contenido del tab
            dcc.Loading(
                id="loading-main", type="dot", color="#6366F1",
                children=html.Div(id="tab-content", style={"position": "relative", "zIndex": "1"}),
            ),
        ], style={"padding": "24px 32px 40px"}),
 
    ], className="main-content", style={
        "marginLeft": "200px",
        "minHeight": "100vh",
        "background": "transparent",
    }),
], style={"fontFamily": "'DM Mono', monospace", "color": "#E2E8F0"})
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
 
# ── Cambiar tab activo ──────────────────────────────────────────────────────
 
@app.callback(
    Output("store-active-tab", "data"),
    Output("header-breadcrumb", "children"),
    Output("header-title", "children"),
    [Input(f"btn-{tab}", "n_clicks") for _, _, tab, _ in NAV_ITEMS],
    prevent_initial_call=True,
)
def cambiar_tab(*args):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
    tab_id = btn_id.replace("btn-", "")
    num, label = next((n, l) for n, l, t, _ in NAV_ITEMS if t == tab_id)
    breadcrumb = f"PLANTA DE PRODUCCIÓN  ›  MÓDULO {num}"
    return tab_id, breadcrumb, label.upper()
 
 
# ── Mostrar / ocultar panel de parámetros ──────────────────────────────────
 
@app.callback(
    Output("params-wrapper", "children"),
    Input("store-active-tab", "data"),
)
def toggle_params(tab):
    # La pestaña de demanda NO tiene parámetros
    if tab == "tab-demanda":
        return html.Div()
    return html.Div([param_panel, html.Div(style={"height": "20px"})])
 
 
# ── Ejecutar pipeline completo ─────────────────────────────────────────────
 
@app.callback(
    Output("store-agr",      "data"),
    Output("store-desag",    "data"),
    Output("store-sim",      "data"),
    Output("store-util",     "data"),
    Output("store-kpis",     "data"),
    Output("store-sensores", "data"),
    Output("store-plan-mes", "data"),
    Output("run-status",     "children"),
    Input("btn-run",   "n_clicks"),
    Input("auto-run",  "n_intervals"),
    State("dd-mes",        "value"),
    State("sl-demanda",    "value"),
    State("sl-horno",      "value"),
    State("chk-opciones",  "value"),
    prevent_initial_call=False,
)
def ejecutar_pipeline(n_clicks, _auto, mes_idx, factor_dem, cap_horno, opciones):
    try:
        # 1. Demanda → H-H
        dem_h = calcular_dem_horas(factor=float(factor_dem or 1.0))
 
        # 2. Planeación agregada
        df_agr, costo = run_agregacion(dem_h)
        prod_hh = dict(zip(df_agr["Mes"], df_agr["Produccion_HH"]))
 
        # 3. Desagregación
        desag = run_desagregacion(prod_hh, float(factor_dem or 1.0))
 
        # 4. Plan del mes seleccionado
        mes_nm = MESES[int(mes_idx or 1)]
        plan_mes = {
            p: int(desag[p].loc[desag[p]["Mes"] == mes_nm, "Produccion"].values[0])
            for p in PRODUCTOS
        }
 
        # 5. Simulación
        cap_rec = {**CAPACIDAD_BASE, "horno": int(cap_horno or 3)}
        factor_t = 0.80 if "turno"  in (opciones or []) else 1.0
        falla    = "falla" in (opciones or [])
        df_l, df_u, df_s = run_simulacion(plan_mes, cap_rec, falla, factor_t)
 
        # 6. Métricas
        df_kpi = calc_kpis(df_l, plan_mes)
 
        # Serializar
        agr_json   = df_agr.to_json()
        desag_json = {p: df.to_json() for p, df in desag.items()}
        sim_json   = df_l.to_json()   if not df_l.empty   else "{}"
        util_json  = df_u.to_json()   if not df_u.empty   else "{}"
        kpi_json   = df_kpi.to_json() if not df_kpi.empty else "{}"
        sen_json   = df_s.to_json()   if not df_s.empty   else "{}"
 
        n_lotes = len(df_l)
        status = (f"✓  Pipeline listo  ·  {n_lotes} lotes  ·  mes: {mes_nm}  ·  "
                  f"costo agregado: COP ${costo:,.0f}")
        return agr_json, desag_json, sim_json, util_json, kpi_json, sen_json, plan_mes, status
 
    except Exception as e:
        import traceback
        return (None,) * 7 + (f"✗ Error: {str(e)}",)
 
 
# ── Correr escenarios what-if ─────────────────────────────────────────────
 
@app.callback(
    Output("store-esc", "data"),
    Input("btn-esc", "n_clicks"),
    State("dd-esc",         "value"),
    State("store-plan-mes", "data"),
    State("store-esc",      "data"),
    prevent_initial_call=True,
)
def correr_escenarios(n, seleccionados, plan_mes, esc_store):
    if not n or not plan_mes or not seleccionados:
        return esc_store or {}
    return correr_escenarios_seleccionados(seleccionados, plan_mes, esc_store or {})
 
 
# ── Renderizar tab activo ─────────────────────────────────────────────────
 
@app.callback(
    Output("tab-content", "children"),
    Input("store-active-tab", "data"),
    State("store-agr",      "data"),
    State("store-desag",    "data"),
    State("store-sim",      "data"),
    State("store-util",     "data"),
    State("store-kpis",     "data"),
    State("store-sensores", "data"),
    State("store-plan-mes", "data"),
    State("store-esc",      "data"),
    State("dd-mes",         "value"),
)
def render_tab(tab, agr_j, desag_j, sim_j, util_j, kpi_j, sen_j, plan_mes, esc_store, mes_idx):
 
    # ════════════════════════════════════════════════════════════
    # 01 · DEMANDA  — solo datos históricos, sin parámetros
    # ════════════════════════════════════════════════════════════
    if tab == "tab-demanda":
        kpis = get_kpis_demanda()
        df_tabla = get_resumen_demanda()
 
        return html.Div([
            page_header("Demanda Histórica", "módulo 01 · análisis de estacionalidad"),
 
            # KPI cards
            dbc.Row([
                dbc.Col(kpi_card("Total Anual",     f"{kpis['total_anual']:,}",  "und",    "#6366F1"), width="auto"),
                dbc.Col(kpi_card("Productos",        kpis["n_productos"],         "",       "#0EA5E9"), width="auto"),
                dbc.Col(kpi_card("Mes Pico",         kpis["mes_pico"],            "",       "#10B981"), width="auto"),
                dbc.Col(kpi_card("Promedio Mensual", f"{kpis['avg_mensual']:,}", "und/mes", "#F59E0B"), width="auto"),
                dbc.Col(kpi_card("H-H Anuales",      f"{kpis['hh_anuales']:,}",  "H-H",    "#EC4899"), width="auto"),
            ], className="g-3 mb-4"),
 
            # Gráficas
            dbc.Row([
                dbc.Col(html.Div([
                    dcc.Graph(
                        figure=fig_barras_demanda(THEME),
                        config={"displayModeBar": False},
                    )
                ], style=CARD), width=8),
                dbc.Col(html.Div([
                    dcc.Graph(
                        figure=fig_heatmap_demanda(THEME),
                        config={"displayModeBar": False},
                    )
                ], style=CARD), width=4),
            ], className="g-3 mb-4"),
 
            section_label("tabla de demanda histórica · unidades por mes"),
            html.Div(tabla_dash(df_tabla, "tbl-demanda"), style=CARD),
        ])
 
    # ════════════════════════════════════════════════════════════
    # 02 · PLANEACIÓN AGREGADA
    # ════════════════════════════════════════════════════════════
    if tab == "tab-agregacion":
        if not agr_j:
            return no_data_msg()
 
        df_agr = pd.read_json(agr_j)
        costo_est = df_agr["Produccion_HH"].sum() * 4310
 
        return html.Div([
            page_header("Planeación Agregada", "módulo 02 · optimización H-H con PuLP / CBC"),
 
            dbc.Row([
                dbc.Col(kpi_card("Producción Total",   f"{df_agr['Produccion_HH'].sum():,.0f}",  "H-H",  "#6366F1"), width="auto"),
                dbc.Col(kpi_card("Demanda Total",       f"{df_agr['Demanda_HH'].sum():,.0f}",    "H-H",  "#0EA5E9"), width="auto"),
                dbc.Col(kpi_card("Backlog Total",        f"{df_agr['Backlog_HH'].sum():,.1f}",    "H-H",  "#EF4444"), width="auto"),
                dbc.Col(kpi_card("Horas Extra",          f"{df_agr['Horas_Extras'].sum():,.1f}",  "H-H",  "#F59E0B"), width="auto"),
                dbc.Col(kpi_card("Contrataciones",       f"{df_agr['Contratacion'].sum():,.0f}",  "",     "#10B981"), width="auto"),
            ], className="g-3 mb-4"),
 
            html.Div([
                dcc.Graph(
                    figure=fig_plan_agregado(df_agr, costo_est, THEME),
                    config={"displayModeBar": False},
                )
            ], style=CARD),
 
            section_label("tabla detallada · plan mensual en H-H"),
            html.Div(tabla_dash(df_agr, "tbl-agr"), style=CARD),
        ])
 
    # ════════════════════════════════════════════════════════════
    # 03 · DESAGREGACIÓN
    # ════════════════════════════════════════════════════════════
    if tab == "tab-desag":
        if not desag_j:
            return no_data_msg()
 
        desag_dict = {p: pd.read_json(v) for p, v in desag_j.items()}
        mes_nm     = MESES[int(mes_idx or 1)]
        total_und  = sum(desag_dict[p]["Produccion"].sum() for p in PRODUCTOS)
 
        return html.Div([
            page_header("Desagregación del Plan", "módulo 03 · unidades por producto y mes"),
 
            dbc.Row([
                dbc.Col(kpi_card("Total Unidades", f"{total_und:,.0f}", "und", "#6366F1"), width="auto"),
                *[
                    dbc.Col(kpi_card(
                        p.replace("_", " "),
                        f"{desag_dict[p]['Produccion'].sum():,.0f}",
                        "und",
                        PROD_COLORS[p],
                    ), width="auto")
                    for p in PRODUCTOS
                ],
            ], className="g-3 mb-4"),
 
            html.Div([
                dcc.Graph(
                    figure=fig_desagregacion(desag_dict, mes_nm, THEME),
                    config={"displayModeBar": False},
                )
            ], style=CARD),
 
            section_label("tabla de producción por producto · mes seleccionado"),
            html.Div([
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Div(p.replace("_", " "), style={
                            "fontSize": "10px", "color": PROD_COLORS[p],
                            "fontFamily": "'Outfit', sans-serif",
                            "fontWeight": "700", "marginBottom": "8px",
                            "letterSpacing": "0.06em",
                        }),
                        tabla_dash(desag_dict[p], f"tbl-desag-{p}", page_size=6),
                    ], style={**CARD, "height": "100%"}), width=6)
                    for i, p in enumerate(PRODUCTOS)
                ], className="g-3"),
            ]),
        ])
 
    # ════════════════════════════════════════════════════════════
    # 04 · SIMULACIÓN
    # ════════════════════════════════════════════════════════════
    if tab == "tab-sim":
        if not sim_j or sim_j == "{}":
            return no_data_msg()
 
        df_l   = pd.read_json(sim_j)
        df_u_r = pd.read_json(util_j) if util_j and util_j != "{}" else pd.DataFrame()
 
        n_lotes  = len(df_l)
        t_total  = round(df_l["t_fin"].max(), 1)       if not df_l.empty else 0
        dur_prom = round(df_l["tiempo_sistema"].mean(), 1) if not df_l.empty else 0
        prods_u  = len(df_l["producto"].unique())       if not df_l.empty else 0
 
        return html.Div([
            page_header("Simulación de Eventos Discretos", "módulo 04 · SimPy · flujo de lotes por recurso"),
 
            dbc.Row([
                dbc.Col(kpi_card("Lotes Simulados",   n_lotes,            "",    "#6366F1"), width="auto"),
                dbc.Col(kpi_card("Tiempo Total",       f"{t_total:,.0f}", "min", "#0EA5E9"), width="auto"),
                dbc.Col(kpi_card("Duración Prom Lote", f"{dur_prom:.1f}", "min", "#10B981"), width="auto"),
                dbc.Col(kpi_card("Productos Activos",  prods_u,            "",   "#F59E0B"), width="auto"),
            ], className="g-3 mb-4"),
 
            html.Div([
                dcc.Graph(
                    figure=fig_gantt(df_l, THEME, n=80),
                    config={"displayModeBar": False},
                )
            ], style=CARD),
 
            html.Div(style={"height": "16px"}),
 
            html.Div([
                dcc.Graph(
                    figure=fig_colas(df_u_r, THEME),
                    config={"displayModeBar": False},
                )
            ], style=CARD),
 
            section_label("registro de lotes simulados · primeros 200"),
            html.Div(
                tabla_dash(
                    df_l.head(200)[["lote_id", "producto", "tamano",
                                    "t_creacion", "t_fin", "tiempo_sistema", "total_espera"]],
                    "tbl-lotes",
                ),
                style=CARD,
            ),
        ])
 
    # ════════════════════════════════════════════════════════════
    # 05 · KPIs & CUELLOS DE BOTELLA
    # ════════════════════════════════════════════════════════════
    if tab == "tab-kpis":
        if not kpi_j or kpi_j == "{}":
            return no_data_msg()
 
        df_kpi = pd.read_json(kpi_j)
        df_u_r = pd.read_json(util_j) if util_j and util_j != "{}" else pd.DataFrame()
        df_ut  = calc_utilizacion(df_u_r)
 
        # KPI resumen rápido
        avg_cumpl = round(df_kpi["Cumplimiento %"].mean(), 1) if not df_kpi.empty else 0
        avg_tp    = round(df_kpi["Throughput (und/h)"].mean(), 2) if not df_kpi.empty else 0
        max_util  = round(df_ut["Utilización_%"].max(), 1) if not df_ut.empty else 0
        cuellos   = int(df_ut["Cuello Botella"].sum()) if not df_ut.empty else 0
 
        return html.Div([
            page_header("KPIs & Cuellos de Botella", "módulo 05 · métricas de desempeño operativo"),
 
            dbc.Row([
                dbc.Col(kpi_card("Cumplimiento Prom",  f"{avg_cumpl}",   "%",      "#10B981"), width="auto"),
                dbc.Col(kpi_card("Throughput Prom",    f"{avg_tp}",      "und/h",  "#6366F1"), width="auto"),
                dbc.Col(kpi_card("Util. Máxima",        f"{max_util}",   "%",      "#F59E0B"), width="auto"),
                dbc.Col(kpi_card("Cuellos de Botella",  cuellos,          "",       "#EF4444"), width="auto"),
            ], className="g-3 mb-4"),
 
            dbc.Row([
                dbc.Col(html.Div([
                    dcc.Graph(
                        figure=fig_kpis_radar(df_kpi, THEME),
                        config={"displayModeBar": False},
                    )
                ], style=CARD), width=6),
                dbc.Col(html.Div([
                    dcc.Graph(
                        figure=fig_utilizacion(df_u_r, THEME),
                        config={"displayModeBar": False},
                    )
                ], style=CARD), width=6),
            ], className="g-3 mb-4"),
 
            section_label("tabla de KPIs por producto"),
            html.Div(tabla_dash(df_kpi, "tbl-kpis", page_size=10), style=CARD),
 
            section_label("utilización de recursos · identificación de cuellos de botella"),
            html.Div(
                tabla_dash(df_ut, "tbl-util", page_size=10) if not df_ut.empty else html.Div(),
                style=CARD,
            ),
        ])
 
    # ════════════════════════════════════════════════════════════
    # 06 · SENSORES VIRTUALES
    # ════════════════════════════════════════════════════════════
    if tab == "tab-sensores":
        if not sen_j or sen_j == "{}":
            return no_data_msg()
 
        df_s = pd.read_json(sen_j)
        t_max  = round(df_s["temperatura"].max(),  1) if not df_s.empty else 0
        t_min  = round(df_s["temperatura"].min(),  1) if not df_s.empty else 0
        t_prom = round(df_s["temperatura"].mean(), 1) if not df_s.empty else 0
        n_lect = len(df_s)
        alertas = int((df_s["temperatura"] > 200).sum()) if not df_s.empty else 0
 
        return html.Div([
            page_header("Sensores Virtuales", "módulo 06 · monitoreo en tiempo real del horno"),
 
            dbc.Row([
                dbc.Col(kpi_card("Temp. Máxima",  f"{t_max}",  "°C", "#EF4444"), width="auto"),
                dbc.Col(kpi_card("Temp. Mínima",  f"{t_min}",  "°C", "#0EA5E9"), width="auto"),
                dbc.Col(kpi_card("Temp. Promedio", f"{t_prom}", "°C", "#F59E0B"), width="auto"),
                dbc.Col(kpi_card("Lecturas",        n_lect,      "",   "#10B981"), width="auto"),
                dbc.Col(kpi_card("Alertas >200°C",  alertas,     "",   "#EC4899"), width="auto"),
            ], className="g-3 mb-4"),
 
            html.Div([
                dcc.Graph(
                    figure=fig_sensores(df_s, THEME),
                    config={"displayModeBar": False},
                )
            ], style=CARD),
 
            section_label("registro de lecturas del sensor"),
            html.Div(
                tabla_dash(df_s.round(2), "tbl-sensores", page_size=12),
                style=CARD,
            ),
        ])
 
    # ════════════════════════════════════════════════════════════
    # 07 · ESCENARIOS WHAT-IF
    # ════════════════════════════════════════════════════════════
    if tab == "tab-escenarios":
        fig_comp = go.Figure()
        if esc_store:
            fig_comp = fig_comparacion_escenarios(esc_store, THEME)
 
        return html.Div([
            page_header("Análisis de Escenarios What-If", "módulo 07 · evaluación comparativa de estrategias"),
 
            # Panel de selección de escenarios
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Span("SELECCIONAR ESCENARIOS", style=LABEL),
                        dcc.Checklist(
                            id="dd-esc",
                            options=ESC_OPTIONS,
                            value=["base", "demanda_20"],
                            inline=True,
                            style={
                                "color": "#CBD5E1", "fontSize": "11px",
                                "fontFamily": "'DM Mono', monospace",
                            },
                            labelStyle={"marginRight": "20px", "display": "inline-flex",
                                        "alignItems": "center", "gap": "6px"},
                        ),
                    ], width=9),
                    dbc.Col([
                        html.Div(style={"height": "20px"}),
                        html.Button(
                            "▶  CORRER ESCENARIOS",
                            id="btn-esc", n_clicks=0,
                            style={
                                "background": "rgba(14,165,233,0.1)",
                                "color": "#0EA5E9",
                                "border": "1px solid rgba(14,165,233,0.35)",
                                "padding": "9px 18px",
                                "fontFamily": "'Outfit', sans-serif",
                                "fontWeight": "700", "fontSize": "12px",
                                "letterSpacing": "0.1em", "borderRadius": "8px",
                                "cursor": "pointer", "whiteSpace": "nowrap",
                            },
                        ),
                    ], width=3, style={"display": "flex", "alignItems": "flex-end"}),
                ], className="g-2"),
            ], style=CARD),
 
            html.Div(style={"height": "16px"}),
 
            html.Div([
                dcc.Graph(id="fig-comp", figure=fig_comp, config={"displayModeBar": False})
            ], style=CARD),
 
            section_label("descripción de escenarios disponibles"),
            html.Div([
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Div(cfg["label"], style={
                            "color": "#A5B4FC", "fontSize": "11px",
                            "fontFamily": "'Outfit', sans-serif",
                            "fontWeight": "700", "marginBottom": "4px",
                        }),
                        html.Div(
                            f"fd={cfg['fd']}  falla={cfg['falla']}  ft={cfg['ft']}  dh={cfg['dh']}  fl={cfg['fl']}",
                            style={"color": "#475569", "fontSize": "9px",
                                   "fontFamily": "'DM Mono', monospace"},
                        ),
                    ], style={**CARD_SM, "borderLeft": "2px solid rgba(99,102,241,0.4)"}), width=4)
                    for k, cfg in ESCENARIOS.items()
                ], className="g-2"),
            ]),
        ])
 
    # Default
    return html.Div("Selecciona una sección.", style={"color": "#475569", "padding": "40px"})
 
 
# ── Actualizar gráfico de comparación cuando cambia el store ───────────────
 
@app.callback(
    Output("fig-comp", "figure"),
    Input("store-esc", "data"),
    prevent_initial_call=True,
)
def update_fig_comp(esc_store):
    if not esc_store:
        return go.Figure()
    return fig_comparacion_escenarios(esc_store, THEME)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print("\n" + "═" * 58)
    print("  GEMELO DIGITAL — DORA DEL HOYO")
    print("  Iniciando servidor en http://127.0.0.1:8050")
    print("═" * 58 + "\n")
    app.run(debug=True, port=8050, host="0.0.0.0")
