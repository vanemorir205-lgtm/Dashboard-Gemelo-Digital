"""
app.py
======
Dashboard principal — Gemelo Digital Dora del Hoyo
===================================================
Integra los módulos:
  datos.py / demanda.py / agregacion.py / desagregacion.py
  simulacion.py / escenarios.py

INSTALACIÓN:
    pip install dash dash-bootstrap-components simpy pulp pandas numpy plotly gunicorn

EJECUCIÓN LOCAL:
    python app.py  →  http://127.0.0.1:8050

RENDER (producción):
    Build command : pip install -r requirements.txt
    Start command : gunicorn app:server --workers 1 --threads 2 --timeout 120
"""

import json as _json
import warnings
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table, ALL
import dash_bootstrap_components as dbc

from datos         import PRODUCTOS, MESES, MESES_C, PROD_COLORS, CAPACIDAD_BASE
from demanda       import get_kpis_demanda, get_resumen_demanda, fig_barras_demanda, fig_heatmap_demanda
from agregacion    import calcular_dem_horas, run_agregacion, fig_plan_agregado
from desagregacion import run_desagregacion, fig_desagregacion
from simulacion    import (run_simulacion, calc_kpis, calc_utilizacion,
                           fig_gantt, fig_colas, fig_utilizacion,
                           fig_kpis_radar, fig_sensores)
from escenarios    import (ESC_OPTIONS, correr_escenarios_seleccionados,
                           fig_comparacion_escenarios)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════
# TEMA
# ═══════════════════════════════════════════════════════

C_BG      = "#F0F4FF"
C_CARD    = "#FFFFFF"
C_SIDEBAR = "#1E1B4B"
C_ACCENT  = "#6366F1"
C_ACCENT2 = "#818CF8"
C_TEAL    = "#0EA5E9"
C_GREEN   = "#10B981"
C_AMBER   = "#F59E0B"
C_PINK    = "#EC4899"
C_RED     = "#EF4444"
C_TEXT    = "#1E1B4B"
C_TEXT2   = "#6B7280"
C_TEXT3   = "#9CA3AF"
C_BORDER  = "#E0E7FF"

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=C_TEXT2, size=11),
    xaxis=dict(gridcolor=C_BORDER, zerolinecolor=C_BORDER, tickfont=dict(size=10, color=C_TEXT3)),
    yaxis=dict(gridcolor=C_BORDER, zerolinecolor=C_BORDER, tickfont=dict(size=10, color=C_TEXT3)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=C_TEXT2),
                orientation="h", y=-0.22, x=0.5, xanchor="center"),
    margin=dict(l=55, r=20, t=55, b=60),
    colorway=[C_ACCENT, C_TEAL, C_GREEN, C_AMBER, C_PINK, "#A78BFA"],
    title=dict(font=dict(size=15, color=C_TEXT,
                         family="Barlow Condensed, Arial Narrow, sans-serif")),
)

S_CARD    = {"background": C_CARD, "border": f"1px solid {C_BORDER}",
             "borderRadius": "12px", "padding": "16px 18px"}
S_LABEL   = {"color": C_TEXT3, "fontSize": "9px", "fontFamily": "IBM Plex Mono, monospace",
             "letterSpacing": "0.12em", "textTransform": "uppercase",
             "marginBottom": "5px", "display": "block"}
S_INPUT   = {"background": "#F8FAFF", "color": C_TEXT, "border": f"1px solid {C_BORDER}",
             "borderRadius": "8px", "fontFamily": "Inter, sans-serif",
             "fontSize": "12px", "padding": "6px 10px"}
S_BTN_PRI = {"background": C_ACCENT, "color": "#FFFFFF", "border": "none",
             "borderRadius": "8px", "fontFamily": "Inter, sans-serif",
             "fontWeight": "600", "fontSize": "13px",
             "padding": "9px 24px", "cursor": "pointer", "width": "100%"}
S_BTN_SEC = {"background": "#EEF2FF", "color": C_ACCENT, "border": f"1px solid #C7D2FE",
             "borderRadius": "8px", "fontFamily": "Inter, sans-serif",
             "fontWeight": "600", "fontSize": "12px",
             "padding": "8px 18px", "cursor": "pointer"}

CUSTOM_CSS = f"""
body {{ background:{C_BG} !important; }} * {{ box-sizing:border-box; }}
::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:{C_BG}; }}
::-webkit-scrollbar-thumb {{ background:#C7D2FE; border-radius:3px; }}
.nav-btn {{ background:transparent; border:none; border-right:3px solid transparent;
            width:100%; text-align:left; padding:9px 18px; cursor:pointer;
            display:flex; align-items:center; transition:background .15s; }}
.nav-btn:hover {{ background:rgba(255,255,255,.07) !important; }}
.nav-btn.active {{ background:rgba(129,140,248,.18) !important;
                   border-right:3px solid {C_ACCENT2} !important; }}
.Select-control {{ background:#F8FAFF !important; border:1px solid {C_BORDER} !important;
                   border-radius:8px !important; font-family:Inter,sans-serif !important;
                   font-size:12px !important; }}
.Select-menu-outer {{ background:{C_CARD} !important; border:1px solid {C_BORDER} !important;
                      border-radius:8px !important; }}
.Select-option {{ color:{C_TEXT2} !important; font-size:12px !important; }}
.Select-option.is-focused {{ background:#EEF2FF !important; color:{C_ACCENT} !important; }}
.rc-slider-track {{ background:{C_ACCENT} !important; }}
.rc-slider-handle {{ border-color:{C_ACCENT} !important; background:white !important;
                     width:16px !important; height:16px !important; margin-top:-6px !important; }}
.rc-slider-rail {{ background:{C_BORDER} !important; }}
.form-check-input:checked {{ background-color:{C_ACCENT} !important; border-color:{C_ACCENT} !important; }}
"""

# ═══════════════════════════════════════════════════════
# COMPONENTES
# ═══════════════════════════════════════════════════════

def kpi_card(label, value, unit="", color=C_ACCENT, pct=75):
    return html.Div([
        html.Span(label, style=S_LABEL),
        html.Div([
            html.Span(str(value), style={"fontSize": "26px", "fontWeight": "700",
                                          "color": color, "fontFamily": "Barlow Condensed, Arial Narrow, sans-serif"}),
            html.Span(f" {unit}", style={"fontSize": "11px", "color": C_TEXT3,
                                          "marginLeft": "4px", "fontFamily": "IBM Plex Mono, monospace"}) if unit else None,
        ]),
        html.Div(style={"height": "3px", "borderRadius": "2px", "background": C_BORDER, "marginTop": "10px"},
                 children=html.Div(style={"height": "100%", "width": f"{min(pct,100)}%",
                                          "borderRadius": "2px", "background": color, "transition": "width .4s ease"})),
    ], style={**S_CARD, "flex": "1", "minWidth": "120px"})


def seccion(titulo, sub=""):
    return html.Div([
        html.H4(titulo, style={"fontFamily": "Barlow Condensed, Arial Narrow, sans-serif",
                                "fontWeight": "700", "fontSize": "18px", "color": C_TEXT, "margin": "0 0 2px 0"}),
        html.P(sub, style={"color": C_TEXT3, "fontSize": "11px", "margin": "0",
                            "fontFamily": "Inter, sans-serif"}) if sub else None,
    ], style={"borderLeft": f"4px solid {C_ACCENT}", "paddingLeft": "12px", "marginBottom": "18px"})


def tabla(df, id_tabla, page_size=12):
    if df is None or df.empty:
        return html.Div("Sin datos.", style={"color": C_TEXT3})
    df_r = df.copy()
    for c in df_r.select_dtypes(include="number").columns:
        df_r[c] = df_r[c].round(2)
    return dash_table.DataTable(
        id=id_tabla,
        columns=[{"name": c, "id": c} for c in df.columns],
        data=df_r.to_dict("records"),
        page_size=page_size,
        style_table={"overflowX": "auto", "borderRadius": "8px"},
        style_header={"backgroundColor": "#F0F4FF", "color": C_ACCENT,
                       "fontFamily": "IBM Plex Mono, monospace", "fontSize": "9px",
                       "border": f"1px solid {C_BORDER}", "letterSpacing": "0.1em",
                       "padding": "8px 12px", "fontWeight": "500"},
        style_cell={"backgroundColor": C_CARD, "color": C_TEXT2,
                     "fontFamily": "Inter, sans-serif", "fontSize": "11px",
                     "border": f"1px solid {C_BORDER}", "padding": "7px 12px", "textAlign": "right"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#F8FAFF"},
            {"if": {"column_id": df.columns[0]},
             "textAlign": "left", "color": C_TEXT, "fontWeight": "500"},
        ],
    )


def no_data():
    return html.Div([
        html.Div("◈", style={"fontSize": "48px", "color": C_BORDER,
                              "textAlign": "center", "paddingTop": "60px"}),
        html.Div("Ejecuta el pipeline primero",
                 style={"textAlign": "center", "color": C_TEXT3, "fontSize": "14px",
                        "fontFamily": "Inter, sans-serif", "padding": "12px 0 60px"}),
    ], style=S_CARD)


# ═══════════════════════════════════════════════════════
# NAVEGACIÓN
# ═══════════════════════════════════════════════════════

NAV_ITEMS = [
    ("01", "Demanda",       "tab-demanda"),
    ("02", "Planeación",    "tab-agregacion"),
    ("03", "Desagregación", "tab-desag"),
    ("04", "Simulación",    "tab-sim"),
    ("05", "KPIs",          "tab-kpis"),
    ("06", "Sensores",      "tab-sensores"),
    ("07", "Escenarios",    "tab-escenarios"),
]


def nav_btn(num, label, tab_id):
    return html.Button(
        [html.Span(num, style={"fontSize": "9px", "marginRight": "10px",
                                "fontFamily": "IBM Plex Mono, monospace",
                                "color": "rgba(255,255,255,.25)"}),
         html.Span(label, style={"fontSize": "12px", "fontFamily": "Inter, sans-serif",
                                  "fontWeight": "500", "color": "rgba(255,255,255,.55)"})],
        id={"type": "nav-btn", "index": tab_id},
        n_clicks=0, className="nav-btn",
    )


# ═══════════════════════════════════════════════════════
# APP DASH
# ═══════════════════════════════════════════════════════

EXTERNAL_CSS = [
    "https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700"
    "&family=IBM+Plex+Mono:wght@300;400;500&family=Inter:wght@400;500;600&display=swap",
    dbc.themes.BOOTSTRAP,
]

app = dash.Dash(__name__, external_stylesheets=EXTERNAL_CSS,
                suppress_callback_exceptions=True,
                title="Gemelo Digital — Dora del Hoyo")
server = app.server   # ← gunicorn / Render

app.index_string = (
    "<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>"
    "{%favicon%}{%css%}<style>" + CUSTOM_CSS + "</style></head>"
    "<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"
)

sidebar = html.Div([
    html.Div([
        html.Div("◈", style={"width": "38px", "height": "38px", "borderRadius": "10px",
                              "background": C_ACCENT, "color": "white", "fontSize": "20px",
                              "display": "flex", "alignItems": "center",
                              "justifyContent": "center", "marginBottom": "10px"}),
        html.Div("DORA DEL HOYO", style={"fontSize": "11px", "fontWeight": "700",
                                          "letterSpacing": "0.14em", "color": "rgba(255,255,255,.9)",
                                          "fontFamily": "Barlow Condensed, Arial Narrow, sans-serif"}),
        html.Div("Gemelo Digital v3", style={"fontSize": "9px", "color": C_ACCENT2,
                                              "fontFamily": "IBM Plex Mono, monospace", "marginTop": "2px"}),
    ], style={"padding": "22px 18px 18px",
              "borderBottom": "1px solid rgba(255,255,255,.07)", "marginBottom": "10px"}),
    html.Div([nav_btn(n, l, t) for n, l, t in NAV_ITEMS]),
    html.Div([
        html.Div(style={"width": "7px", "height": "7px", "borderRadius": "50%",
                        "background": C_GREEN, "marginRight": "8px"}),
        html.Span("Sistema activo", style={"fontSize": "9px", "color": "rgba(255,255,255,.2)",
                                           "fontFamily": "IBM Plex Mono, monospace"}),
    ], style={"position": "absolute", "bottom": "0", "width": "100%",
              "padding": "12px 18px", "borderTop": "1px solid rgba(255,255,255,.06)",
              "display": "flex", "alignItems": "center"}),
], style={"width": "200px", "minWidth": "200px", "minHeight": "100vh",
          "background": C_SIDEBAR, "display": "flex", "flexDirection": "column",
          "position": "fixed", "top": "0", "left": "0", "zIndex": "200"})

panel_config = html.Div([
    dbc.Row([
        dbc.Col([html.Span("MES A SIMULAR", style=S_LABEL),
                 dcc.Dropdown(id="dd-mes",
                              options=[{"label": m, "value": i} for i, m in enumerate(MESES)],
                              value=2, clearable=False, style={**S_INPUT, "padding": "0"})], width=2),
        dbc.Col([html.Span("FACTOR DEMANDA", style=S_LABEL),
                 dcc.Slider(id="sl-demanda", min=0.5, max=2.0, step=0.1, value=1.0,
                            marks={0.5: "0.5×", 1.0: "1×", 1.5: "1.5×", 2.0: "2×"},
                            tooltip={"placement": "top", "always_visible": True})], width=3),
        dbc.Col([html.Span("ESTACIONES HORNO", style=S_LABEL),
                 dcc.Slider(id="sl-horno", min=1, max=6, step=1, value=3,
                            marks={i: str(i) for i in range(1, 7)},
                            tooltip={"placement": "top", "always_visible": True})], width=2),
        dbc.Col([html.Span("OPCIONES", style=S_LABEL),
                 dbc.Checklist(id="chk-opciones",
                               options=[{"label": "Falla en horno", "value": "falla"},
                                        {"label": "Doble turno (−20%)", "value": "turno"}],
                               value=[], switch=True)], width=3),
        dbc.Col([html.Div(id="run-status",
                          style={"color": C_GREEN, "fontSize": "10px",
                                 "fontFamily": "IBM Plex Mono, monospace",
                                 "marginBottom": "6px", "minHeight": "15px"}),
                 html.Button("▶  Ejecutar pipeline", id="btn-run", n_clicks=0, style=S_BTN_PRI)], width=2),
    ], align="start", className="g-3"),
], style={**S_CARD, "margin": "12px 22px 0"})

app.layout = html.Div([
    dcc.Store(id="store-active-tab", data="tab-demanda"),
    dcc.Store(id="store-agr",        data=None),
    dcc.Store(id="store-desag",      data=None),
    dcc.Store(id="store-sim",        data=None),
    dcc.Store(id="store-util",       data=None),
    dcc.Store(id="store-kpis",       data=None),
    dcc.Store(id="store-sensores",   data=None),
    dcc.Store(id="store-plan-mes",   data=None),
    dcc.Store(id="store-esc",        data={}),
    dcc.Location(id="url", refresh=False),
    sidebar,
    html.Div([
        html.Div([
            html.Div([
                html.Div([html.Span("● ", style={"color": C_ACCENT, "fontSize": "10px"}),
                          html.Span("PLANTA DE PRODUCCIÓN  ·  GEMELO DIGITAL",
                                    style={"fontFamily": "IBM Plex Mono, monospace",
                                           "fontSize": "9px", "letterSpacing": "0.16em", "color": C_TEXT3})],
                         style={"marginBottom": "4px"}),
                html.Div(id="header-tab-name", children="Demanda",
                         style={"fontFamily": "Barlow Condensed, Arial Narrow, sans-serif",
                                "fontWeight": "700", "fontSize": "28px", "color": C_TEXT}),
            ]),
            html.Div([
                html.Span("PuLP / CBC", style={"background": "#EEF2FF", "color": "#4338CA",
                                               "border": "1px solid #C7D2FE", "fontSize": "9px",
                                               "padding": "4px 10px", "borderRadius": "20px",
                                               "fontFamily": "IBM Plex Mono, monospace", "marginRight": "6px"}),
                html.Span("SimPy DES", style={"background": "#F0FDFA", "color": "#0F766E",
                                              "border": "1px solid #99F6E4", "fontSize": "9px",
                                              "padding": "4px 10px", "borderRadius": "20px",
                                              "fontFamily": "IBM Plex Mono, monospace", "marginRight": "6px"}),
                html.Span("● En línea", style={"background": "#F0FDF4", "color": "#15803D",
                                               "border": "1px solid #BBF7D0", "fontSize": "9px",
                                               "padding": "4px 10px", "borderRadius": "20px",
                                               "fontFamily": "IBM Plex Mono, monospace"}),
            ]),
        ], style={"padding": "16px 22px", "background": C_CARD,
                  "borderBottom": f"1px solid {C_BORDER}",
                  "display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
        panel_config,
        html.Hr(style={"borderColor": C_BORDER, "margin": "14px 0"}),
        dcc.Loading(id="loading-main", type="circle", color=C_ACCENT,
                    children=html.Div(id="tab-content", style={"padding": "0 22px 40px"})),
    ], style={"marginLeft": "200px", "minHeight": "100vh", "background": C_BG}),
], style={"fontFamily": "Inter, sans-serif", "color": C_TEXT})


# ═══════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════

@app.callback(
    Output("store-active-tab", "data"),
    Output("header-tab-name",  "children"),
    Input({"type": "nav-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def cambiar_tab(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    tab_id = _json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
    label  = next(l for _, l, t in NAV_ITEMS if t == tab_id)
    return tab_id, label


@app.callback(
    Output("store-agr",      "data"),
    Output("store-desag",    "data"),
    Output("store-sim",      "data"),
    Output("store-util",     "data"),
    Output("store-kpis",     "data"),
    Output("store-sensores", "data"),
    Output("store-plan-mes", "data"),
    Output("run-status",     "children"),
    Input("btn-run",         "n_clicks"),
    State("dd-mes",          "value"),
    State("sl-demanda",      "value"),
    State("sl-horno",        "value"),
    State("chk-opciones",    "value"),
    prevent_initial_call=True,
)
def ejecutar_pipeline(n, mes_idx, factor_dem, cap_horno, opciones):
    if not n:
        return (None,) * 7 + ("",)
    try:
        dem_h         = calcular_dem_horas(factor_dem)
        df_agr, costo = run_agregacion(dem_h)
        prod_hh       = dict(zip(df_agr["Mes"], df_agr["Produccion_HH"]))
        desag         = run_desagregacion(prod_hh, factor_dem)
        mes_nm        = MESES[mes_idx]
        plan_mes      = {p: int(desag[p].loc[desag[p]["Mes"] == mes_nm, "Produccion"].values[0])
                         for p in PRODUCTOS}
        cap_rec       = {**CAPACIDAD_BASE, "horno": int(cap_horno)}
        factor_t      = 0.80 if "turno" in (opciones or []) else 1.0
        falla         = "falla" in (opciones or [])
        df_l, df_u, df_s = run_simulacion(plan_mes, cap_rec, falla, factor_t)
        df_kpi        = calc_kpis(df_l, plan_mes)

        return (
            df_agr.to_json(),
            {p: df.to_json() for p, df in desag.items()},
            df_l.to_json()   if not df_l.empty   else "{}",
            df_u.to_json()   if not df_u.empty   else "{}",
            df_kpi.to_json() if not df_kpi.empty else "{}",
            df_s.to_json()   if not df_s.empty   else "{}",
            plan_mes,
            f"✓  {len(df_l)} lotes  ·  {mes_nm}  ·  COP ${costo / 1e6:.1f}M",
        )
    except Exception as e:
        return (None,) * 7 + (f"✗ Error: {str(e)}",)


@app.callback(
    Output("store-esc",     "data"),
    Input("btn-esc",        "n_clicks"),
    State("dd-esc",         "value"),
    State("store-plan-mes", "data"),
    State("store-esc",      "data"),
    prevent_initial_call=True,
)
def cb_escenarios(n, esc_sel, plan_mes, esc_store):
    if not n or not plan_mes or not esc_sel:
        return esc_store or {}
    return correr_escenarios_seleccionados(esc_sel, plan_mes, esc_store)


@app.callback(
    Output("tab-content", "children"),
    Input("store-active-tab", "data"),
    Input("store-agr",        "data"),
    Input("store-desag",      "data"),
    Input("store-sim",        "data"),
    Input("store-util",       "data"),
    Input("store-kpis",       "data"),
    Input("store-sensores",   "data"),
    Input("store-esc",        "data"),
    Input("url",              "pathname"),
    State("store-plan-mes",   "data"),
    State("dd-mes",           "value"),
)
def render_tab(tab, agr_j, desag_j, sim_j, util_j, kpi_j, sen_j,
               esc_store, _url, plan_mes, mes_idx):

    if not tab:
        tab = "tab-demanda"
    GC = {"displayModeBar": False, "responsive": True}

    if tab == "tab-demanda":
        kpis = get_kpis_demanda()
        return html.Div([
            seccion("Demanda histórica", "análisis de estacionalidad por producto"),
            dbc.Row([
                dbc.Col(kpi_card("Total anual",  f"{kpis['total_anual']:,}", "und",   C_ACCENT, 100), width=2),
                dbc.Col(kpi_card("Productos",     kpis["n_productos"],       "líneas",C_TEAL,   100), width=2),
                dbc.Col(kpi_card("Mes pico",      kpis["mes_pico"],          "",      C_GREEN,   80), width=2),
                dbc.Col(kpi_card("Avg mensual",   f"{kpis['avg_mensual']:,}","und",   C_AMBER,   65), width=2),
                dbc.Col(kpi_card("H-H anuales",   f"{kpis['hh_anuales']:,}","H-H",  C_PINK,    55), width=2),
            ], className="g-3 mb-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_barras_demanda(PLOTLY_THEME),  config=GC), width=7),
                dbc.Col(dcc.Graph(figure=fig_heatmap_demanda(PLOTLY_THEME), config=GC), width=5),
            ], className="g-3 mb-4"),
            seccion("Tabla de demanda histórica", "unidades por mes"),
            tabla(get_resumen_demanda(), "tbl-dem"),
        ])

    if tab == "tab-agregacion":
        if not agr_j: return no_data()
        df_agr = pd.read_json(agr_j)
        costo  = df_agr["Produccion_HH"].sum() * 4310
        return html.Div([
            seccion("Planeación agregada", "optimización en horas-hombre — PuLP / CBC"),
            dbc.Row([
                dbc.Col(kpi_card("Prod. total H-H", f"{df_agr['Produccion_HH'].sum():,.0f}", "H-H", C_ACCENT, 100), width=2),
                dbc.Col(kpi_card("Backlog total",    f"{df_agr['Backlog_HH'].sum():,.1f}",    "H-H", C_RED,    40),  width=2),
                dbc.Col(kpi_card("H. extra total",   f"{df_agr['Horas_Extras'].sum():,.1f}",  "H-H", C_AMBER,  55),  width=2),
                dbc.Col(kpi_card("Contrataciones",   f"{df_agr['Contratacion'].sum():,.0f}",  "",    C_GREEN,  40),  width=2),
                dbc.Col(kpi_card("Costo óptimo",     f"${costo / 1e6:.1f}M",                  "COP", C_TEAL,   70),  width=2),
            ], className="g-3 mb-4"),
            dcc.Graph(figure=fig_plan_agregado(df_agr, costo, PLOTLY_THEME), config=GC),
            html.Div(style={"height": "14px"}),
            seccion("Tabla detallada", "plan mensual en horas-hombre"),
            tabla(df_agr, "tbl-agr"),
        ])

    if tab == "tab-desag":
        if not desag_j: return no_data()
        dd     = {p: pd.read_json(v) for p, v in desag_j.items()}
        mes_nm = MESES[mes_idx or 2]
        total  = sum(dd[p]["Produccion"].sum() for p in PRODUCTOS)
        return html.Div([
            seccion("Desagregación del plan", "unidades por producto y mes"),
            dbc.Row([
                dbc.Col(kpi_card("Total unidades", f"{total:,.0f}", "und", C_ACCENT, 100), width=2),
                *[dbc.Col(kpi_card(p.replace("_", " ")[:12],
                                   f"{dd[p]['Produccion'].sum():,.0f}", "und", PROD_COLORS[p],
                                   int(dd[p]["Produccion"].sum() / total * 100) if total > 0 else 0),
                          width=2) for p in PRODUCTOS],
            ], className="g-3 mb-4"),
            dcc.Graph(figure=fig_desagregacion(dd, mes_nm, PLOTLY_THEME), config=GC),
        ])

    if tab == "tab-sim":
        if not sim_j or sim_j == "{}": return no_data()
        df_l = pd.read_json(sim_j)
        if df_l.empty: return no_data()
        df_u = pd.read_json(util_j) if util_j and util_j != "{}" else pd.DataFrame()
        return html.Div([
            seccion("Simulación de eventos discretos", "SimPy — flujo de lotes por recurso"),
            dbc.Row([
                dbc.Col(kpi_card("Lotes simulados",  len(df_l),                              "",    C_ACCENT, 100), width=2),
                dbc.Col(kpi_card("Tiempo total",      f"{df_l['t_fin'].max():,.0f}",          "min", C_TEAL,    80), width=2),
                dbc.Col(kpi_card("Dur. prom. lote",   f"{df_l['tiempo_sistema'].mean():.1f}", "min", C_GREEN,   65), width=2),
                dbc.Col(kpi_card("Espera prom.",      f"{df_l['total_espera'].mean():.1f}",   "min", C_AMBER,   50), width=2),
            ], className="g-3 mb-4"),
            dcc.Graph(figure=fig_gantt(df_l, PLOTLY_THEME), config=GC),
            html.Div(style={"height": "12px"}),
            dcc.Graph(figure=fig_colas(df_u, PLOTLY_THEME), config=GC),
            html.Div(style={"height": "12px"}),
            seccion("Registro de lotes", "primeros 200 resultados"),
            tabla(df_l.head(200)[["lote_id", "producto", "tamano",
                                   "t_creacion", "t_fin", "tiempo_sistema", "total_espera"]], "tbl-lotes"),
        ])

    if tab == "tab-kpis":
        if not kpi_j or kpi_j == "{}": return no_data()
        df_kpi = pd.read_json(kpi_j)
        if df_kpi.empty: return no_data()
        df_ut  = pd.read_json(util_j) if util_j and util_j != "{}" else pd.DataFrame()
        dfut2  = calc_utilizacion(df_ut)
        cum    = round(df_kpi["Cumplimiento %"].mean(), 1)
        tp     = round(df_kpi["Throughput (und/h)"].mean(), 2)
        nb     = int(dfut2["Cuello Botella"].sum()) if not dfut2.empty else 0
        return html.Div([
            seccion("KPIs & cuellos de botella", "métricas de desempeño operativo"),
            dbc.Row([
                dbc.Col(kpi_card("Cumplimiento prom.", f"{cum}", "%",
                                  C_GREEN if cum >= 95 else C_AMBER, int(cum)), width=2),
                dbc.Col(kpi_card("Throughput prom.",   f"{tp}", "und/h", C_TEAL, 70),   width=2),
                dbc.Col(kpi_card("Recursos analizados", len(dfut2), "",  C_ACCENT, 100), width=2),
                dbc.Col(kpi_card("Cuellos de botella",  nb, "",
                                  C_RED if nb > 0 else C_GREEN,
                                  int(nb / max(len(dfut2), 1) * 100)), width=2),
            ], className="g-3 mb-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_kpis_radar(df_kpi, PLOTLY_THEME), config=GC), width=6),
                dbc.Col(dcc.Graph(figure=fig_utilizacion(df_ut, PLOTLY_THEME),  config=GC), width=6),
            ], className="g-3 mb-4"),
            seccion("Tabla KPIs por producto"),
            tabla(df_kpi, "tbl-kpis", page_size=10),
            html.Div(style={"height": "16px"}),
            seccion("Utilización de recursos"),
            tabla(dfut2, "tbl-util", page_size=10) if not dfut2.empty else html.Div(),
        ])

    if tab == "tab-sensores":
        if not sen_j or sen_j == "{}": return no_data()
        df_s = pd.read_json(sen_j)
        if df_s.empty: return no_data()
        t_max   = round(df_s["temperatura"].max(), 1)
        t_min   = round(df_s["temperatura"].min(), 1)
        t_avg   = round(df_s["temperatura"].mean(), 1)
        alertas = int((df_s["temperatura"] > 200).sum())
        return html.Div([
            seccion("Sensores virtuales", "monitoreo en tiempo real — horno de producción"),
            dbc.Row([
                dbc.Col(kpi_card("Temp. máxima",    t_max,      "°C", C_PINK,  int(t_max / 250 * 100)), width=2),
                dbc.Col(kpi_card("Temp. mínima",    t_min,      "°C", C_TEAL,  int(t_min / 250 * 100)), width=2),
                dbc.Col(kpi_card("Temp. promedio",  t_avg,      "°C", C_AMBER, int(t_avg / 250 * 100)), width=2),
                dbc.Col(kpi_card("Lecturas totales", len(df_s), "",   C_ACCENT, 100),                    width=2),
                dbc.Col(kpi_card("Alertas >200°C",  alertas,   "",
                                  C_RED if alertas > 0 else C_GREEN,
                                  int(alertas / max(len(df_s), 1) * 100)), width=2),
            ], className="g-3 mb-4"),
            dcc.Graph(figure=fig_sensores(df_s, PLOTLY_THEME), config=GC),
        ])

    if tab == "tab-escenarios":
        fig_comp = (fig_comparacion_escenarios(esc_store, PLOTLY_THEME)
                    if esc_store else go.Figure())
        return html.Div([
            seccion("Análisis de escenarios what-if", "evaluación comparativa de estrategias"),
            html.Div([
                html.Span("SELECCIONAR ESCENARIOS", style=S_LABEL),
                dcc.Checklist(id="dd-esc", options=ESC_OPTIONS,
                              value=["base", "demanda_20"], inline=True,
                              labelStyle={"marginRight": "20px", "display": "inline-block"}),
                html.Div(style={"height": "12px"}),
                html.Button("▶  Correr escenarios seleccionados",
                            id="btn-esc", n_clicks=0, style=S_BTN_SEC),
            ], style={**S_CARD, "marginBottom": "16px"}),
            dcc.Graph(id="fig-comp", figure=fig_comp, config=GC),
        ])

    return html.Div("Selecciona una sección.", style={"color": C_TEXT3, "padding": "40px"})


@app.callback(
    Output("fig-comp", "figure"),
    Input("store-esc", "data"),
    prevent_initial_call=True,
)
def update_comp(esc_store):
    return fig_comparacion_escenarios(esc_store, PLOTLY_THEME) if esc_store else go.Figure()


# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  GEMELO DIGITAL v3 — DORA DEL HOYO")
    print("  → http://127.0.0.1:8050")
    print("═" * 60 + "\n")
    app.run(debug=False, port=8050, host="0.0.0.0")
