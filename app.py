"""
dashboard_gemelo.py — v3.0
===========================
Gemelo Digital — Dora del Hoyo

CAMBIOS v3.0:
  ▸ Pestañas completamente separadas (una cosa por pestaña)
  ▸ Demanda: datos históricos fijos, sin factor de alteración
  ▸ Agregación: solo costos, fuerza laboral y parámetros técnicos
  ▸ Desagregación: solo costo producción e inventario
  ▸ Simulación: parámetros operativos configurables
  ▸ Escenarios: configuración visible + botón ejecutar
  ▸ KPIs: KPIs base arriba, escenarios abajo, sensores al final

INSTALACIÓN:
    pip install dash dash-bootstrap-components simpy pulp pandas numpy plotly

EJECUCIÓN LOCAL:
    python dashboard_gemelo.py  →  http://127.0.0.1:8050
"""

import math, random, warnings
import numpy as np
import pandas as pd
import simpy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, PULP_CBC_CMD
import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# DATOS MAESTROS — FIJOS, NO SE ALTERAN DESDE UI
# ═══════════════════════════════════════════════════════════════════════════════
PRODUCTOS = ["Brownies", "Mantecadas", "Mantecadas_Amapola", "Torta_Naranja", "Pan_Maiz"]
MESES = ["January","February","March","April","May","June",
         "July","August","September","October","November","December"]

DEM_HISTORICA = {
    "Brownies":           [315,804,734,541,494, 59,315,803,734,541,494, 59],
    "Mantecadas":         [125,780,432,910,275, 68,512,834,690,455,389,120],
    "Mantecadas_Amapola": [320,710,520,251,631,150,330,220,710,610,489,180],
    "Torta_Naranja":      [100,250,200,101,190, 50,100,220,200,170,180,187],
    "Pan_Maiz":           [330,140,143, 73, 83, 48, 70, 89,118, 83, 67, 87],
}
HORAS_PRODUCTO = {
    "Brownies":0.866,"Mantecadas":0.175,"Mantecadas_Amapola":0.175,
    "Torta_Naranja":0.175,"Pan_Maiz":0.312,
}
INV_INICIAL = {p: 0 for p in PRODUCTOS}

RUTAS = {
    "Brownies":          [("Mezclado","mezcla",12,18),("Moldeado","dosificado",8,14),
                          ("Horneado","horno",30,40),("Enfriamiento","enfriamiento",25,35),
                          ("Corte_Empaque","empaque",8,12)],
    "Mantecadas":        [("Mezclado","mezcla",12,18),("Dosificado","dosificado",16,24),
                          ("Horneado","horno",20,30),("Enfriamiento","enfriamiento",35,55),
                          ("Empaque","empaque",4,6)],
    "Mantecadas_Amapola":[("Mezclado","mezcla",12,18),("Inc_Semillas","mezcla",8,12),
                          ("Dosificado","dosificado",16,24),("Horneado","horno",20,30),
                          ("Enfriamiento","enfriamiento",36,54),("Empaque","empaque",4,6)],
    "Torta_Naranja":     [("Mezclado","mezcla",16,24),("Dosificado","dosificado",8,12),
                          ("Horneado","horno",32,48),("Enfriamiento","enfriamiento",48,72),
                          ("Desmolde","dosificado",8,12),("Empaque","empaque",8,12)],
    "Pan_Maiz":          [("Mezclado","mezcla",12,18),("Amasado","amasado",16,24),
                          ("Moldeado","dosificado",12,18),("Horneado","horno",28,42),
                          ("Enfriamiento","enfriamiento",36,54),("Empaque","empaque",4,6)],
}
TAMANO_LOTE_BASE = {
    "Brownies":12,"Mantecadas":10,"Mantecadas_Amapola":10,"Torta_Naranja":12,"Pan_Maiz":15
}
CAPACIDAD_BASE = {
    "mezcla":2,"dosificado":2,"horno":3,"enfriamiento":4,"empaque":2,"amasado":1
}
PROD_COLORS = {
    "Brownies":"#E8A838","Mantecadas":"#4FC3F7","Mantecadas_Amapola":"#81C784",
    "Torta_Naranja":"#CE93D8","Pan_Maiz":"#FF8A65"
}

# Demanda fija en H-H (calculada UNA vez, no varía desde la UI)
DEM_HORAS_FIJA = {
    m: round(sum(DEM_HISTORICA[p][i] * HORAS_PRODUCTO[p] for p in PRODUCTOS), 4)
    for i, m in enumerate(MESES)
}

# Plan fijo de unidades (mes representativo: promedio anual por producto)
PLAN_UNIDADES_BASE = {
    p: int(sum(DEM_HISTORICA[p]) / 12) for p in PRODUCTOS
}

# ═══════════════════════════════════════════════════════════════════════════════
# MOTORES DE CÁLCULO
# ═══════════════════════════════════════════════════════════════════════════════

def run_agregacion(params):
    """
    Planeación agregada usando DEM_HORAS_FIJA (demanda no modificable desde UI).
    Solo los parámetros de costos y capacidad cambian.
    """
    Ct   = params.get("Ct",   4310)
    Ht   = params.get("Ht",   100000)
    PIt  = params.get("PIt",  100000)
    CRt  = params.get("CRt",  11364)
    COt  = params.get("COt",  14205)
    Wm   = params.get("CW_mas",  14204)
    Wd   = params.get("CW_menos",15061)
    M    = params.get("M",    1)
    LRi  = params.get("LR_inicial", 44 * 4 * 10)
    dw   = params.get("delta_trabajadores", 50)

    mdl = LpProblem("Agr_DoraDelHoyo", LpMinimize)
    P      = LpVariable.dicts("P",   MESES, lowBound=0)
    Iv     = LpVariable.dicts("I",   MESES, lowBound=0)
    S      = LpVariable.dicts("S",   MESES, lowBound=0)
    LR     = LpVariable.dicts("LR",  MESES, lowBound=0)
    LO     = LpVariable.dicts("LO",  MESES, lowBound=0)
    LU     = LpVariable.dicts("LU",  MESES, lowBound=0)
    NI     = LpVariable.dicts("NI",  MESES)
    Wmas   = LpVariable.dicts("Wm",  MESES, lowBound=0)
    Wmenos = LpVariable.dicts("Wd",  MESES, lowBound=0)

    mdl += lpSum(
        Ct*P[t] + Ht*Iv[t] + PIt*S[t] +
        CRt*LR[t] + COt*LO[t] +
        Wm*Wmas[t] + Wd*Wmenos[t]
        for t in MESES
    )

    for idx, t in enumerate(MESES):
        d  = DEM_HORAS_FIJA[t]
        tp = MESES[idx-1] if idx > 0 else None

        if idx == 0: mdl += NI[t] == P[t] - d
        else:        mdl += NI[t] == NI[tp] + P[t] - d

        mdl += NI[t] == Iv[t] - S[t]
        mdl += LU[t] + LO[t] == M * P[t]
        mdl += LU[t] <= LR[t]

        if idx == 0: mdl += LR[t] == LRi  + Wmas[t] - Wmenos[t]
        else:        mdl += LR[t] == LR[tp]+ Wmas[t] - Wmenos[t]

        mdl += Wmas[t]   <= dw
        mdl += Wmenos[t] <= dw

    mdl.solve(PULP_CBC_CMD(msg=False))
    costo = value(mdl.objective) or 0

    ini_l, fin_l = [], []
    for idx, t in enumerate(MESES):
        ini = 0.0 if idx == 0 else fin_l[-1]
        ini_l.append(ini)
        fin_l.append(ini + (P[t].varValue or 0) - DEM_HORAS_FIJA[t])

    df = pd.DataFrame({
        "Mes":                   MESES,
        "Demanda_HH":            [round(DEM_HORAS_FIJA[t], 2) for t in MESES],
        "Produccion_HH":         [round(P[t].varValue  or 0, 2) for t in MESES],
        "Backlog_HH":            [round(S[t].varValue  or 0, 2) for t in MESES],
        "Horas_Regulares":       [round(LR[t].varValue or 0, 2) for t in MESES],
        "Horas_Extras":          [round(LO[t].varValue or 0, 2) for t in MESES],
        "Inventario_Inicial_HH": [round(v, 2) for v in ini_l],
        "Inventario_Final_HH":   [round(v, 2) for v in fin_l],
        "Contratacion":          [round(Wmas[t].varValue   or 0, 2) for t in MESES],
        "Despidos":              [round(Wmenos[t].varValue or 0, 2) for t in MESES],
    })
    return df, costo


def run_desagregacion(prod_hh, cost_prod=1.0, cost_inv=1.0):
    """Desagregación con parámetros de costo configurables. Demanda fija."""
    mdl = LpProblem("Desag_DoraDelHoyo", LpMinimize)
    X  = {(p,t): LpVariable(f"X_{p}_{t}", lowBound=0) for p in PRODUCTOS for t in MESES}
    Iv = {(p,t): LpVariable(f"I_{p}_{t}", lowBound=0) for p in PRODUCTOS for t in MESES}
    Sv = {(p,t): LpVariable(f"S_{p}_{t}", lowBound=0) for p in PRODUCTOS for t in MESES}

    mdl += lpSum(
        cost_inv * Iv[p,t] + cost_prod * 10000 * Sv[p,t]
        for p in PRODUCTOS for t in MESES
    )

    for idx, t in enumerate(MESES):
        tp = MESES[idx-1] if idx > 0 else None
        mdl += lpSum(HORAS_PRODUCTO[p] * X[p,t] for p in PRODUCTOS) <= prod_hh[t]
        for p in PRODUCTOS:
            d = DEM_HISTORICA[p][idx]  # demanda fija, sin factor
            if idx == 0:
                mdl += Iv[p,t] - Sv[p,t] == INV_INICIAL[p] + X[p,t] - d
            else:
                mdl += Iv[p,t] - Sv[p,t] == Iv[p,tp] - Sv[p,tp] + X[p,t] - d

    mdl.solve(PULP_CBC_CMD(msg=False))

    out = {}
    for p in PRODUCTOS:
        rows = []
        for idx, t in enumerate(MESES):
            xv  = round(X[p,t].varValue  or 0, 2)
            iv  = round(Iv[p,t].varValue or 0, 2)
            sv  = round(Sv[p,t].varValue or 0, 2)
            ini = INV_INICIAL[p] if idx == 0 else round(Iv[p,MESES[idx-1]].varValue or 0, 2)
            rows.append({
                "Mes": t,
                "Demanda":    DEM_HISTORICA[p][idx],
                "Produccion": xv,
                "Inv_Ini":    ini,
                "Inv_Fin":    iv,
                "Backlog":    sv,
            })
        out[p] = pd.DataFrame(rows)
    return out


def run_simulacion(plan_unidades, cap_recursos=None, falla=False, factor_t=1.0,
                   tamano_lote=None, semilla=42, tiempos_custom=None):
    random.seed(semilla); np.random.seed(semilla)
    if cap_recursos is None: cap_recursos = CAPACIDAD_BASE.copy()
    if tamano_lote  is None: tamano_lote  = TAMANO_LOTE_BASE.copy()

    rutas_ef = {}
    for prod, etapas in RUTAS.items():
        rutas_ef[prod] = [
            (eta, rec,
             tiempos_custom[rec][0] if tiempos_custom and rec in tiempos_custom else tmin,
             tiempos_custom[rec][1] if tiempos_custom and rec in tiempos_custom else tmax)
            for eta, rec, tmin, tmax in etapas
        ]

    lotes_data, uso_rec, sensores = [], [], []

    def reg_uso(env, recursos):
        ts = round(env.now, 3)
        for nm, r in recursos.items():
            uso_rec.append({"tiempo":ts,"recurso":nm,"ocupados":r.count,
                            "cola":len(r.queue),"capacidad":r.capacity})

    def sensor_horno(env, recursos):
        while True:
            ocp = recursos["horno"].count
            sensores.append({
                "tiempo":      round(env.now, 1),
                "temperatura": round(np.random.normal(160 + ocp*20, 5), 2),
                "horno_ocup":  ocp,
                "horno_cola":  len(recursos["horno"].queue),
            })
            yield env.timeout(10)

    def proceso_lote(env, lid, prod, tam, recursos):
        t0 = env.now
        for eta, rec_nm, tmin, tmax in rutas_ef[prod]:
            escala = math.sqrt(tam / TAMANO_LOTE_BASE[prod])
            tp_proc = random.uniform(tmin, tmax) * escala * factor_t
            if falla and rec_nm == "horno":
                tp_proc += random.uniform(10, 30)
            with recursos[rec_nm].request() as req:
                yield req
                reg_uso(env, recursos)
                yield env.timeout(tp_proc)
            reg_uso(env, recursos)
        lotes_data.append({
            "lote_id":        lid,
            "producto":       prod,
            "tamano":         tam,
            "t_creacion":     round(t0, 3),
            "t_fin":          round(env.now, 3),
            "tiempo_sistema": round(env.now - t0, 3),
            "total_espera":   round(env.now - t0 - sum(
                random.uniform(tmin,tmax)*math.sqrt(tam/TAMANO_LOTE_BASE[prod])*factor_t
                for _,_,tmin,tmax in rutas_ef[prod]
            ), 3),
        })

    env      = simpy.Environment()
    recursos = {nm: simpy.Resource(env, capacity=cap) for nm, cap in cap_recursos.items()}
    env.process(sensor_horno(env, recursos))

    dur_mes = 44 * 4 * 60
    lotes = []; ctr = [0]
    for prod, unid in plan_unidades.items():
        if unid <= 0: continue
        tam  = tamano_lote[prod]
        n    = math.ceil(unid / tam)
        tasa = dur_mes / max(n, 1)
        ta   = random.expovariate(1/max(tasa, 1))
        rem  = unid
        for _ in range(n):
            lotes.append((round(ta, 2), prod, min(tam, int(rem))))
            rem -= tam
            ta  += random.expovariate(1/max(tasa, 1))
    lotes.sort(key=lambda x: x[0])

    def lanzador():
        for ta, prod, tam in lotes:
            yield env.timeout(max(ta - env.now, 0))
            lid = f"{prod[:3].upper()}_{ctr[0]:04d}"; ctr[0] += 1
            env.process(proceso_lote(env, lid, prod, tam, recursos))
    env.process(lanzador())
    env.run(until=dur_mes * 1.8)

    return (
        pd.DataFrame(lotes_data) if lotes_data else pd.DataFrame(),
        pd.DataFrame(uso_rec)    if uso_rec    else pd.DataFrame(),
        pd.DataFrame(sensores)   if sensores   else pd.DataFrame(),
    )


def calc_utilizacion(df_u):
    if df_u.empty: return pd.DataFrame()
    filas = []
    for rec, grp in df_u.groupby("recurso"):
        grp = grp.sort_values("tiempo"); cap = grp["capacidad"].iloc[0]
        t = grp["tiempo"].values; ocp = grp["ocupados"].values
        if len(t) > 1 and (t[-1] - t[0]) > 0:
            fn   = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            util = round(fn(ocp, t) / (cap*(t[-1]-t[0])) * 100, 2)
        else:
            util = 0.0
        filas.append({
            "Recurso":        rec,
            "Utilización_%":  util,
            "Cola Prom":      round(grp["cola"].mean(), 3),
            "Cola Máx":       int(grp["cola"].max()),
            "Capacidad":      int(cap),
            "Cuello Botella": util >= 80 or grp["cola"].mean() > 0.5,
        })
    return pd.DataFrame(filas).sort_values("Utilización_%", ascending=False).reset_index(drop=True)


def calc_kpis(df_l, plan):
    if df_l.empty: return pd.DataFrame()
    dur  = (df_l["t_fin"].max() - df_l["t_creacion"].min()) / 60
    rows = []
    for p in PRODUCTOS:
        sub = df_l[df_l["producto"] == p]
        if sub.empty: continue
        und      = sub["tamano"].sum()
        plan_und = plan.get(p, 0)
        tp       = round(und / max(dur, 0.01), 3)
        lt       = round(sub["tiempo_sistema"].mean(), 3)
        rows.append({
            "Producto":              p,
            "Und Producidas":        und,
            "Plan":                  plan_und,
            "Throughput (und/h)":    tp,
            "Cycle Time (min/und)":  round((sub["tiempo_sistema"]/sub["tamano"]).mean(), 3),
            "Lead Time (min/lote)":  lt,
            "WIP Prom":              round(tp*(lt/60), 2),
            "Takt Time (min/lote)":  round((44*4*60)/max((sum(DEM_HISTORICA[p])/12)/TAMANO_LOTE_BASE[p],1),2),
            "Cumplimiento %":        round(min(und/max(plan_und,1)*100, 100), 2),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURAS PLOTLY
# ═══════════════════════════════════════════════════════════════════════════════
THEME = dict(
    paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
    font=dict(family="IBM Plex Mono, monospace", color="#212529", size=11),
    xaxis=dict(gridcolor="#e9ecef", zerolinecolor="#dee2e6"),
    yaxis=dict(gridcolor="#e9ecef", zerolinecolor="#dee2e6"),
    legend=dict(bgcolor="rgba(255,255,255,0.8)", font=dict(size=10)),
    margin=dict(l=50, r=20, t=55, b=50),
)

def _apply_theme(fig, title="", height=400):
    fig.update_layout(
        **THEME,
        title=dict(text=title, x=0.5,
                   font=dict(size=15, color="#E8A838",
                             family="Barlow Condensed, sans-serif")),
        height=height,
    )
    return fig


def fig_demanda_barras():
    fig = go.Figure()
    for p in PRODUCTOS:
        fig.add_trace(go.Bar(
            x=MESES, y=DEM_HISTORICA[p],
            name=p.replace("_"," "),
            marker_color=PROD_COLORS[p], opacity=0.85,
        ))
    _apply_theme(fig, "Demanda Histórica por Producto (unidades/mes)", 420)
    fig.update_layout(
        barmode="group", xaxis_title="Mes", yaxis_title="Unidades",
        legend=dict(orientation="h", y=-0.28, x=0.5, xanchor="center"),
    )
    return fig


def fig_demanda_heatmap():
    z   = [[DEM_HISTORICA[p][i] for i in range(12)] for p in PRODUCTOS]
    fig = go.Figure(go.Heatmap(
        z=z, x=MESES,
        y=[p.replace("_"," ") for p in PRODUCTOS],
        colorscale="YlOrBr",
        hovertemplate="%{y}<br>%{x}<br>%{z} unidades<extra></extra>",
    ))
    return _apply_theme(fig, "Mapa de Calor — Estacionalidad", 360)


def fig_demanda_lineas():
    fig = go.Figure()
    for p in PRODUCTOS:
        fig.add_trace(go.Scatter(
            x=MESES, y=DEM_HISTORICA[p],
            mode="lines+markers",
            name=p.replace("_"," "),
            line=dict(color=PROD_COLORS[p], width=2),
            marker=dict(size=6),
        ))
    _apply_theme(fig, "Tendencia Mensual por Producto", 380)
    fig.update_layout(
        xaxis_title="Mes", yaxis_title="Unidades",
        legend=dict(orientation="h", y=-0.28, x=0.5, xanchor="center"),
    )
    return fig


def fig_agregacion(df_agr, costo):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_agr["Mes"], y=df_agr["Inventario_Inicial_HH"],
        name="Inv. Inicial (H-H)", marker_color="#5C6BC0", opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        x=df_agr["Mes"], y=df_agr["Produccion_HH"],
        name="Producción (H-H)", marker_color="#E8A838", opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=df_agr["Mes"], y=df_agr["Demanda_HH"],
        mode="lines+markers", name="Demanda (H-H)",
        line=dict(color="#81C784", dash="dash", width=2.5),
        marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=df_agr["Mes"], y=df_agr["Horas_Regulares"],
        mode="lines", name="Cap. Regular",
        line=dict(color="#FF8A65", dash="dot", width=2),
    ))
    _apply_theme(fig, f"Plan Agregado — Costo Óptimo: COP ${costo:,.0f}", 420)
    fig.update_layout(
        barmode="stack", xaxis_title="Mes", yaxis_title="Horas-Hombre",
        legend=dict(orientation="h", y=-0.28, x=0.5, xanchor="center"),
    )
    return fig


def fig_fuerza_laboral(df_agr):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Contrataciones vs Despidos", "Horas Regulares vs Extras"])
    fig.add_trace(go.Bar(x=df_agr["Mes"], y=df_agr["Contratacion"],
                         name="Contrataciones", marker_color="#81C784"), row=1, col=1)
    fig.add_trace(go.Bar(x=df_agr["Mes"], y=df_agr["Despidos"],
                         name="Despidos", marker_color="#FF8A65"), row=1, col=1)
    fig.add_trace(go.Bar(x=df_agr["Mes"], y=df_agr["Horas_Regulares"],
                         name="Regulares", marker_color="#4FC3F7"), row=1, col=2)
    fig.add_trace(go.Bar(x=df_agr["Mes"], y=df_agr["Horas_Extras"],
                         name="Extras", marker_color="#CE93D8"), row=1, col=2)
    _apply_theme(fig, "Dinámica de Fuerza Laboral", 380)
    fig.update_layout(barmode="group")
    return fig


def fig_desagregacion_grid(desag_dict):
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[p.replace("_"," ") for p in PRODUCTOS],
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )
    for idx, p in enumerate(PRODUCTOS):
        r, c = idx//2+1, idx%2+1
        df   = desag_dict[p]
        fig.add_trace(go.Bar(
            x=df["Mes"], y=df["Produccion"],
            name=p, showlegend=False,
            marker_color=PROD_COLORS[p], opacity=0.85,
        ), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=df["Mes"], y=df["Demanda"],
            mode="lines+markers", showlegend=False,
            line=dict(color="#81C784", dash="dash", width=1.5),
            marker=dict(size=5),
        ), row=r, col=c)
    _apply_theme(fig, "Desagregación por Producto — Producción vs Demanda (unidades/mes)", 720)
    return fig


def fig_gantt(df_l, n=80):
    if df_l.empty: return go.Figure()
    sub = df_l.head(n).copy()
    fig = go.Figure()
    for _, row in sub.iterrows():
        fig.add_trace(go.Bar(
            x=[row["tiempo_sistema"]], y=[row["lote_id"]],
            base=[row["t_creacion"]], orientation="h",
            marker_color=PROD_COLORS.get(row["producto"],"#aaa"),
            opacity=0.8, showlegend=False,
            hovertemplate=(
                f"<b>{row['producto']}</b><br>"
                f"Inicio: {row['t_creacion']:.0f} min<br>"
                f"Duración: {row['tiempo_sistema']:.1f} min<extra></extra>"
            ),
        ))
    for p, c in PROD_COLORS.items():
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=c,
                             name=p.replace("_"," "), showlegend=True))
    _apply_theme(fig, "Diagrama de Gantt — Lotes de Producción", max(350, len(sub)*7))
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Tiempo (min)", yaxis_title="Lote ID",
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
    )
    return fig


def fig_colas(df_u):
    if df_u.empty: return go.Figure()
    fig  = go.Figure()
    pal  = ["#E8A838","#4FC3F7","#81C784","#CE93D8","#FF8A65","#F06292"]
    for i, (rec, grp) in enumerate(df_u.groupby("recurso")):
        grp = grp.sort_values("tiempo")
        fig.add_trace(go.Scatter(
            x=grp["tiempo"], y=grp["cola"],
            mode="lines", name=rec,
            line=dict(color=pal[i%len(pal)], width=1.5),
        ))
    _apply_theme(fig, "Evolución de Colas por Recurso", 380)
    fig.update_xaxes(title_text="Tiempo (min)")
    fig.update_yaxes(title_text="Cola")
    fig.update_layout(
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
    )
    return fig


def fig_utilizacion_gauge(df_ut):
    if df_ut.empty: return []
    figs = []
    for _, row in df_ut.iterrows():
        val   = row["Utilización_%"]
        color = "#00cc96" if val < 80 else "#ffa726" if val < 95 else "#ef5350"
        fig   = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            number={"suffix":"%","font":{"size":36}},
            gauge={
                "axis":{"range":[0,100],"visible":False},
                "bar":{"color":color,"thickness":1.0},
                "bgcolor":"#f8f9fa","borderwidth":0,"shape":"angular",
            },
            title={"text":row["Recurso"],"font":{"size":14}},
        ))
        fig.update_layout(height=160, margin=dict(l=5,r=5,t=30,b=10), template="plotly_white")
        figs.append(fig)
    return figs


def fig_kpis_radar(df_kpi):
    if df_kpi.empty: return go.Figure()
    cats = ["Throughput (und/h)","Cycle Time (min/und)","Lead Time (min/lote)",
            "WIP Prom","Cumplimiento %"]
    fig  = go.Figure()
    for _, row in df_kpi.iterrows():
        maxv = [max(df_kpi[c].max(), 0.01) for c in cats]
        norm = [round(row.get(c,0)/m*100,1) for c,m in zip(cats,maxv)]
        norm.append(norm[0])
        fig.add_trace(go.Scatterpolar(
            r=norm, theta=cats+[cats[0]], fill="toself", opacity=0.6,
            name=row["Producto"].replace("_"," "),
            line=dict(color=PROD_COLORS.get(row["Producto"],"#aaa"), width=2),
        ))
    _apply_theme(fig, "Radar de KPIs por Producto (normalizado)", 400)
    fig.update_layout(
        polar=dict(bgcolor="#f8f9fa",
                   radialaxis=dict(visible=True, gridcolor="#dee2e6"),
                   angularaxis=dict(gridcolor="#dee2e6")),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    )
    return fig


def fig_sensores(df_s):
    if df_s.empty: return go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Temperatura Horno (°C)","Ocupación Horno"])
    fig.add_trace(go.Scatter(
        x=df_s["tiempo"], y=df_s["temperatura"],
        mode="lines", name="Temp",
        line=dict(color="#FF8A65", width=1.5),
    ), row=1, col=1)
    fig.add_hline(y=200, line_dash="dash", line_color="#c0392b",
                  annotation_text="Límite 200°C", row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_s["tiempo"], y=df_s["horno_ocup"],
        mode="lines", fill="tozeroy",
        fillcolor="rgba(79,195,247,0.12)",
        line=dict(color="#4FC3F7", width=1.5), name="Ocup.",
    ), row=2, col=1)
    _apply_theme(fig, "Sensores Virtuales — Monitor del Horno", 440)
    fig.update_xaxes(title_text="Tiempo (min)", row=2, col=1)
    return fig


def fig_comparacion_escenarios(resultados_esc):
    if not resultados_esc: return go.Figure()
    filas = []
    for nm, res in resultados_esc.items():
        dk = res.get("kpis", pd.DataFrame())
        du = res.get("util", pd.DataFrame())
        if dk.empty: continue
        fila = {"Escenario": nm}
        for col in ["Throughput (und/h)","Lead Time (min/lote)","WIP Prom","Cumplimiento %"]:
            if col in dk.columns: fila[col] = round(dk[col].mean(), 2)
        if not du.empty and "Utilización_%" in du.columns:
            fila["Util Máx %"] = round(du["Utilización_%"].max(), 2)
        filas.append(fila)
    df = pd.DataFrame(filas)
    metricas = [
        ("Throughput (und/h)",    "Throughput (und/h)"),
        ("Lead Time (min/lote)",  "Lead Time (min)"),
        ("Cumplimiento %",        "Cumplimiento (%)"),
        ("Util Máx %",            "Util. Máx (%)"),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[m[1] for m in metricas])
    pal = ["#E8A838","#4FC3F7","#81C784","#CE93D8","#FF8A65","#F06292"]
    for i, (col, _) in enumerate(metricas):
        r, c = i//2+1, i%2+1
        if col not in df.columns: continue
        fig.add_trace(go.Bar(
            x=df["Escenario"], y=df[col], showlegend=False,
            marker_color=[pal[j%len(pal)] for j in range(len(df))],
            text=df[col].apply(lambda v: f"{v:.2f}"), textposition="outside",
        ), row=r, col=c)
    _apply_theme(fig, "Comparación de Escenarios What-If", 500)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# ESTILOS
# ═══════════════════════════════════════════════════════════════════════════════
CARD_S  = {"background":"#ffffff","border":"1px solid #dee2e6",
           "borderRadius":"6px","padding":"16px"}
LABEL_S = {"color":"#6c757d","fontSize":"10px",
           "fontFamily":"IBM Plex Mono, monospace",
           "letterSpacing":"0.12em","textTransform":"uppercase","marginBottom":"4px"}
INP_S   = {"background":"#f0f2f5","color":"#212529","border":"1px solid #ced4da",
           "borderRadius":"4px","fontFamily":"IBM Plex Mono, monospace","fontSize":"12px"}

def _lbl(text):
    return html.Div(text, style=LABEL_S)

def _num(id_, val, step=1.0, mn=None):
    kw = {"id":id_,"value":val,"step":step,"style":INP_S,"debounce":True,"type":"number"}
    if mn is not None: kw["min"] = mn
    return dbc.Input(**kw)

def _sl(id_, lo, hi, val, step=1, marks=None):
    return dcc.Slider(id=id_, min=lo, max=hi, step=step, value=val,
                      marks=marks or {lo:str(lo), hi:str(hi)},
                      tooltip={"placement":"top","always_visible":True})

def kpi_card(titulo, valor, unidad="", color="#E8A838", icon="◈"):
    return html.Div([
        html.Div(icon+" "+titulo, style={**LABEL_S,"color":"#6c757d"}),
        html.Div([
            html.Span(str(valor),
                      style={"fontSize":"26px","fontWeight":"600","color":color,
                             "fontFamily":"Barlow Condensed, sans-serif"}),
            html.Span(" "+unidad,
                      style={"fontSize":"11px","color":"#6c757d","marginLeft":"4px"}),
        ]),
    ], style={**CARD_S,"minWidth":"140px"})

def sec_titulo(texto, sub=""):
    return html.Div([
        html.H4(texto, style={"fontFamily":"Barlow Condensed, sans-serif","fontWeight":"700",
                               "color":"#E8A838","margin":"0 0 2px 0","fontSize":"20px",
                               "letterSpacing":"0.06em"}),
        html.P(sub, style={"color":"#6c757d","fontSize":"11px","margin":"0",
                            "fontFamily":"IBM Plex Mono, monospace"}) if sub else None,
    ], style={"borderLeft":"3px solid #E8A838","paddingLeft":"12px","marginBottom":"20px"})

def tabla_dash(df, id_t, page_size=12):
    if df is None or df.empty:
        return html.Div("Sin datos", style={"color":"#6c757d"})
    return dash_table.DataTable(
        id=id_t,
        columns=[{"name":c,"id":c} for c in df.columns],
        data=df.round(3).to_dict("records"),
        page_size=page_size,
        style_table={"overflowX":"auto"},
        style_header={"backgroundColor":"#f8f9fa","color":"#E8A838","fontSize":"10px",
                       "fontFamily":"IBM Plex Mono, monospace","border":"1px solid #dee2e6",
                       "letterSpacing":"0.08em"},
        style_cell={"backgroundColor":"#ffffff","color":"#343a40","fontSize":"11px",
                     "fontFamily":"IBM Plex Mono, monospace","border":"1px solid #dee2e6",
                     "padding":"6px 10px","textAlign":"right"},
        style_data_conditional=[{"if":{"row_index":"odd"},"backgroundColor":"#f8f9fa"}],
    )

def _no_data(msg="Ejecuta el pipeline primero (botón ▶ en la pestaña correspondiente)"):
    return html.Div([
        html.Div("◈", style={"fontSize":"48px","color":"#dee2e6","textAlign":"center","paddingTop":"40px"}),
        html.Div(msg, style={"textAlign":"center","color":"#495057","fontSize":"13px",
                              "fontFamily":"IBM Plex Mono, monospace","marginTop":"12px"}),
    ])

def _btn(id_, texto, color_bg="#E8A838", color_txt="#0a0d11"):
    return html.Button(texto, id=id_, n_clicks=0, style={
        "background":color_bg,"color":color_txt,"border":"none",
        "padding":"10px 28px","fontFamily":"Barlow Condensed, sans-serif",
        "fontWeight":"700","fontSize":"13px","letterSpacing":"0.1em",
        "borderRadius":"4px","cursor":"pointer","width":"100%",
    })

def _status(id_):
    return html.Div(id=id_, style={"color":"#198754","fontSize":"12px",
                                    "fontFamily":"IBM Plex Mono, monospace","marginTop":"8px"})

# ═══════════════════════════════════════════════════════════════════════════════
# TABS — CONTENIDO ESTÁTICO DE CONFIGURACIÓN POR PESTAÑA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Tab 1: Demanda ── sin controles, datos fijos
TAB_DEMANDA_CTRL = html.Div([
    html.Div([
        html.Span("ℹ️ ", style={"fontSize":"16px"}),
        html.Span("La demanda es histórica y fija. No se puede modificar desde la interfaz.",
                  style={"fontSize":"12px","fontFamily":"IBM Plex Mono, monospace","color":"#6c757d"}),
    ], style={**CARD_S,"marginBottom":"16px","background":"#f0f6ff",
              "borderColor":"#b8d4f8"}),
])

# ── Tab 2: Planeación Agregada ── costos + fuerza laboral + técnico
TAB_AGRE_CTRL = html.Div([
    html.Div([
        dbc.Accordion([

            dbc.AccordionItem(title="💰 Costos Operativos", children=[
                dbc.Row([
                    dbc.Col([_lbl("COSTO PRODUCCIÓN (Ct) COP/H-H"), _num("agr-ct",  4310,  100)], width=3),
                    dbc.Col([_lbl("COSTO INVENTARIO (Ht) COP/H-H"), _num("agr-ht",  100000, 5000)], width=3),
                    dbc.Col([_lbl("PENALIZACIÓN BACKLOG (PIt)"),     _num("agr-pit", 100000, 5000)], width=3),
                    dbc.Col([_lbl("STOCK DE SEGURIDAD (fracc.)"),    _num("agr-invseg", 0.0, 0.01, mn=0.0)], width=3),
                ], className="g-3"),
            ], item_id="acc-costos"),

            dbc.AccordionItem(title="👷 Fuerza Laboral", children=[
                dbc.Row([
                    dbc.Col([_lbl("COSTO HORA REGULAR (CRt)"),  _num("agr-crt",     11364,  100)], width=3),
                    dbc.Col([_lbl("COSTO HORA EXTRA (COt)"),    _num("agr-cot",     14205,  100)], width=3),
                    dbc.Col([_lbl("COSTO CONTRATAR (CW+)"),     _num("agr-cwmas",   14204,  100)], width=3),
                    dbc.Col([_lbl("COSTO DESPEDIR (CW−)"),      _num("agr-cwmenos", 15061,  100)], width=3),
                ], className="g-3 mb-3"),
                dbc.Row([
                    dbc.Col([_lbl("TRABAJADORES INICIALES"),    _num("agr-trab",    10,  1, mn=1)], width=3),
                    dbc.Col([_lbl("HORAS/SEMANA"),              _num("agr-hsem",    44,  1, mn=1)], width=3),
                    dbc.Col([_lbl("SEMANAS/MES"),               _num("agr-nsem",    4,   1, mn=1)], width=3),
                    dbc.Col([_lbl("MÁX CONTRAT/DESPIDO"),       _num("agr-maxdw",   50,  1, mn=0)], width=3),
                ], className="g-3"),
            ], item_id="acc-laboral"),

            dbc.AccordionItem(title="⚙️ Parámetros Técnicos", children=[
                dbc.Row([
                    dbc.Col([_lbl("H-H POR UNIDAD AGREGADA (M)"), _num("agr-m", 1.0, 0.1, mn=0.1)], width=4),
                    dbc.Col([
                        _lbl("EFICIENCIA OPERATIVA (%)"),
                        _sl("agr-ef", 50, 110, 95, 1, {50:"50",80:"80",110:"110"}),
                    ], width=4),
                    dbc.Col([
                        _lbl("AUSENTISMO (%)"),
                        _sl("agr-aus", 0, 30, 5, 1, {0:"0",15:"15",30:"30"}),
                    ], width=4),
                ], className="g-3 mb-3"),
                html.Div(id="agr-cap-display",
                         style={"background":"rgba(0,123,255,0.06)","border":"1px solid rgba(0,123,255,0.2)",
                                 "borderRadius":"8px","padding":"10px","textAlign":"center",
                                 "fontFamily":"IBM Plex Mono, monospace","fontSize":"12px",
                                 "color":"#007bff","marginTop":"4px"}),
            ], item_id="acc-tecnico"),

        ], active_item="acc-costos", always_open=True, style={"border":"none"}),

        html.Div(style={"height":"12px"}),
        _btn("btn-agr", "▶  CALCULAR PLAN AGREGADO"),
        _status("agr-status"),
    ], style=CARD_S),
])

# ── Tab 3: Desagregación ── solo costos de prod e inventario
TAB_DESAG_CTRL = html.Div([
    html.Div([
        html.P("Requiere haber calculado el plan agregado primero.",
               style={"color":"#6c757d","fontSize":"11px","fontFamily":"IBM Plex Mono, monospace",
                       "marginBottom":"16px"}),
        dbc.Row([
            dbc.Col([
                _lbl("COSTO DE PRODUCCIÓN (relativo)"),
                _num("desag-cprod", 1.0, 0.1, mn=0.1),
                html.Div("Mayor valor → produce justo a tiempo",
                         style={"fontSize":"10px","color":"#6c757d","fontFamily":"IBM Plex Mono, monospace",
                                 "marginTop":"4px"}),
            ], width=4),
            dbc.Col([
                _lbl("COSTO DE INVENTARIO (relativo)"),
                _num("desag-cinv", 1.0, 0.1, mn=0.1),
                html.Div("Mayor valor → minimiza inventario acumulado",
                         style={"fontSize":"10px","color":"#6c757d","fontFamily":"IBM Plex Mono, monospace",
                                 "marginTop":"4px"}),
            ], width=4),
            dbc.Col([
                html.Div([
                    html.Div("💡 Guía de parámetros", style={**LABEL_S,"fontSize":"11px","color":"#E8A838"}),
                    html.Div("Ht > Ct → produce justo a tiempo",
                             style={"fontSize":"11px","color":"#00796B","fontFamily":"IBM Plex Mono, monospace"}),
                    html.Div("Ct > Ht → minimiza producción total",
                             style={"fontSize":"11px","color":"#D32F2F","fontFamily":"IBM Plex Mono, monospace"}),
                    html.Div("Ht = Ct → balance",
                             style={"fontSize":"11px","color":"#495057","fontFamily":"IBM Plex Mono, monospace"}),
                ], style={**CARD_S,"background":"#f8f9fa","padding":"10px"}),
            ], width=4),
        ], className="g-3 mb-3"),
        _btn("btn-desag", "▶  CALCULAR DESAGREGACIÓN"),
        _status("desag-status"),
    ], style=CARD_S),
])

# ── Tab 4: Simulación ── parámetros operativos
TAB_SIM_CTRL = html.Div([
    html.Div([
        html.P("Requiere haber calculado la desagregación primero.",
               style={"color":"#6c757d","fontSize":"11px","fontFamily":"IBM Plex Mono, monospace",
                       "marginBottom":"16px"}),

        dbc.Accordion([

            dbc.AccordionItem(title="🏭 Parámetros Operativos", children=[
                dbc.Row([
                    dbc.Col([_lbl("MES A SIMULAR"),
                             dcc.Dropdown(id="sim-mes",
                                          options=[{"label":m,"value":i} for i,m in enumerate(MESES)],
                                          value=1, clearable=False, style=INP_S)], width=3),
                    dbc.Col([_lbl("FACTOR TIEMPO (1=normal, 0.8=doble turno)"),
                             _sl("sim-ft", 0.5, 1.5, 1.0, 0.05, {0.5:"0.5",1.0:"1",1.5:"1.5"})], width=3),
                    dbc.Col([_lbl("OPCIONES"),
                             dbc.Checklist(id="sim-opciones",
                                           options=[
                                               {"label":"Simular falla en horno","value":"falla"},
                                               {"label":"Doble turno (−20% tiempos)","value":"turno"},
                                           ],
                                           value=[], switch=True,
                                           style={"fontSize":"12px","fontFamily":"IBM Plex Mono, monospace"})
                             ], width=3),
                    dbc.Col([_lbl("SEMILLA ALEATORIA"),
                             _num("sim-semilla", 42, 1, mn=0)], width=3),
                ], className="g-3"),
            ], item_id="acc-simop"),

            dbc.AccordionItem(title="🔧 Capacidad por Estación (nº equipos)", children=[
                dbc.Row([
                    dbc.Col([_lbl("MEZCLA"),       _num("sim-nmezcla", 2, 1, mn=1)], width=2),
                    dbc.Col([_lbl("DOSIFICADO"),   _num("sim-ndosif",  2, 1, mn=1)], width=2),
                    dbc.Col([_lbl("HORNO"),        _num("sim-nhorno",  3, 1, mn=1)], width=2),
                    dbc.Col([_lbl("ENFRIAMIENTO"), _num("sim-nenfr",   4, 1, mn=1)], width=2),
                    dbc.Col([_lbl("EMPAQUE"),      _num("sim-nempaq",  2, 1, mn=1)], width=2),
                    dbc.Col([_lbl("AMASADO"),      _num("sim-namasad", 1, 1, mn=1)], width=2),
                ], className="g-3"),
            ], item_id="acc-simcap"),

            dbc.AccordionItem(title="⏱️ Tiempos por Estación (min mín – máx)", children=[
                dbc.Row([
                    dbc.Col([_lbl("MEZCLA"),
                             dcc.RangeSlider(id="sim-t-mezcla", min=1, max=60, step=1,
                                             value=[12,18], tooltip={"always_visible":True})], width=4),
                    dbc.Col([_lbl("DOSIFICADO"),
                             dcc.RangeSlider(id="sim-t-dosif", min=1, max=60, step=1,
                                             value=[8,24], tooltip={"always_visible":True})], width=4),
                    dbc.Col([_lbl("HORNO"),
                             dcc.RangeSlider(id="sim-t-horno", min=5, max=120, step=1,
                                             value=[20,48], tooltip={"always_visible":True})], width=4),
                ], className="g-3 mb-3"),
                dbc.Row([
                    dbc.Col([_lbl("ENFRIAMIENTO"),
                             dcc.RangeSlider(id="sim-t-enfr", min=5, max=120, step=1,
                                             value=[25,72], tooltip={"always_visible":True})], width=4),
                    dbc.Col([_lbl("EMPAQUE"),
                             dcc.RangeSlider(id="sim-t-empaq", min=1, max=30, step=1,
                                             value=[4,12], tooltip={"always_visible":True})], width=4),
                    dbc.Col([_lbl("AMASADO"),
                             dcc.RangeSlider(id="sim-t-amasad", min=5, max=60, step=1,
                                             value=[16,24], tooltip={"always_visible":True})], width=4),
                ], className="g-3"),
            ], item_id="acc-simtiempos"),

        ], active_item="acc-simop", always_open=True, style={"border":"none"}),

        html.Div(style={"height":"12px"}),
        _btn("btn-sim", "▶  EJECUTAR SIMULACIÓN", "#1e2a3a", "#4FC3F7"),
        _status("sim-status"),
    ], style=CARD_S),
])

# ── Tab 5: KPIs ── sin controles de configuración
TAB_KPI_CTRL = html.Div([
    html.Div([
        html.Span("ℹ️ ", style={"fontSize":"16px"}),
        html.Span("Los KPIs se calculan automáticamente a partir de la simulación y los escenarios.",
                  style={"fontSize":"12px","fontFamily":"IBM Plex Mono, monospace","color":"#6c757d"}),
    ], style={**CARD_S,"background":"#f0f6ff","borderColor":"#b8d4f8"}),
])

# ── Tab 6: Sensores ── sin controles
TAB_SENS_CTRL = html.Div([
    html.Div([
        html.Span("ℹ️ ", style={"fontSize":"16px"}),
        html.Span("Datos de sensores virtuales generados durante la última simulación.",
                  style={"fontSize":"12px","fontFamily":"IBM Plex Mono, monospace","color":"#6c757d"}),
    ], style={**CARD_S,"background":"#f0f6ff","borderColor":"#b8d4f8"}),
])

# ── Tab 7: Escenarios ── selección de escenarios
TAB_ESC_CTRL = html.Div([
    html.Div([
        html.P("Selecciona los escenarios a comparar. Cada escenario varía los parámetros indicados.",
               style={"color":"#6c757d","fontSize":"11px","fontFamily":"IBM Plex Mono, monospace",
                       "marginBottom":"16px"}),
        dcc.Checklist(
            id="esc-sels",
            options=[
                {"label":"Base (parámetros actuales)",           "value":"base"},
                {"label":"Demanda +20% (factor demanda × 1.2)",  "value":"demanda_20"},
                {"label":"Falla de horno (+10-30 min por lote)", "value":"falla_horno"},
                {"label":"Reducir cap. horno (−1 equipo)",       "value":"red_cap"},
                {"label":"Doble turno (−20% tiempos proceso)",   "value":"doble_turno"},
                {"label":"Lotes +50% (tamaño de lote × 1.5)",    "value":"lote_grande"},
                {"label":"Optimizado (+1 horno, −15% tiempos)",  "value":"optimizado"},
            ],
            value=["base","demanda_20"],
            style={"fontSize":"13px","fontFamily":"IBM Plex Mono, monospace"},
            labelStyle={"display":"block","marginBottom":"8px"},
        ),
        html.Div(style={"height":"12px"}),
        _btn("btn-esc", "▶  CORRER ESCENARIOS", "#1e2a3a", "#4FC3F7"),
        _status("esc-status"),
    ], style=CARD_S),
])

CTRL_POR_TAB = {
    "tab-demanda":    TAB_DEMANDA_CTRL,
    "tab-agregacion": TAB_AGRE_CTRL,
    "tab-desag":      TAB_DESAG_CTRL,
    "tab-sim":        TAB_SIM_CTRL,
    "tab-kpis":       TAB_KPI_CTRL,
    "tab-sensores":   TAB_SENS_CTRL,
    "tab-escenarios": TAB_ESC_CTRL,
}

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR Y LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
NAV_ITEMS = [
    ("01","DEMANDA",       "tab-demanda"),
    ("02","PLANEACIÓN",    "tab-agregacion"),
    ("03","DESAGREGACIÓN", "tab-desag"),
    ("04","SIMULACIÓN",    "tab-sim"),
    ("05","KPIs",          "tab-kpis"),
    ("06","SENSORES",      "tab-sensores"),
    ("07","ESCENARIOS",    "tab-escenarios"),
]

sidebar = html.Div([
    html.Div([
        html.Div("◈", style={"fontSize":"28px","color":"#E8A838","lineHeight":"1"}),
        html.Div("DORA DEL HOYO",
                 style={"fontSize":"11px","fontWeight":"700","letterSpacing":"0.18em",
                        "color":"#212529","fontFamily":"Barlow Condensed, sans-serif"}),
        html.Div("GEMELO DIGITAL",
                 style={"fontSize":"9px","color":"#E8A838","letterSpacing":"0.22em",
                        "fontFamily":"IBM Plex Mono, monospace"}),
    ], style={"padding":"24px 16px 20px","borderBottom":"1px solid #dee2e6","marginBottom":"12px"}),

    html.Div([
        html.Button(
            [html.Span(n, style={"fontSize":"9px","color":"#E8A838","marginRight":"8px",
                                  "fontFamily":"IBM Plex Mono, monospace"}),
             html.Span(l, style={"fontSize":"12px","letterSpacing":"0.1em"})],
            id=f"btn-nav-{t}", n_clicks=0,
            style={"background":"transparent","border":"none","color":"#6c757d",
                   "width":"100%","textAlign":"left","padding":"10px 16px","cursor":"pointer",
                   "fontFamily":"Barlow Condensed, sans-serif","fontWeight":"600"},
        )
        for n, l, t in NAV_ITEMS
    ]),
], style={"width":"200px","minHeight":"100vh","background":"#f8f9fa",
          "borderRight":"1px solid #dee2e6","display":"flex","flexDirection":"column",
          "position":"fixed","top":"0","left":"0","zIndex":"100"})

stores = html.Div([
    dcc.Store(id="store-active-tab",  data="tab-demanda"),
    dcc.Store(id="store-agr",         data=None),
    dcc.Store(id="store-desag",       data=None),
    dcc.Store(id="store-sim",         data=None),
    dcc.Store(id="store-util",        data=None),
    dcc.Store(id="store-kpis-base",   data=None),
    dcc.Store(id="store-sens",        data=None),
    dcc.Store(id="store-plan-mes",    data=None),
    dcc.Store(id="store-esc",         data={}),
])

app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700&family=IBM+Plex+Mono:wght@300;400;500&display=swap",
        dbc.themes.BOOTSTRAP,
    ],
    suppress_callback_exceptions=True,
    title="Gemelo Digital — Dora del Hoyo",
)
server = app.server

app.layout = html.Div([
    stores,
    sidebar,
    html.Div([
        # Header
        html.Div([
            html.Div([
                html.Span("●", style={"color":"#E8A838","marginRight":"8px"}),
                html.Span("PLANTA DE PRODUCCIÓN — GEMELO DIGITAL",
                          style={"fontFamily":"IBM Plex Mono, monospace","fontSize":"11px",
                                 "letterSpacing":"0.18em","color":"#6c757d"}),
            ]),
            html.Div(id="header-tab-name",
                     style={"fontFamily":"Barlow Condensed, sans-serif","fontWeight":"700",
                             "fontSize":"28px","color":"#212529","letterSpacing":"0.04em"}),
        ], style={"padding":"20px 24px 16px","borderBottom":"1px solid #dee2e6",
                   "background":"#f8f9fa"}),

        # Panel de configuración específico por tab
        html.Div(id="config-panel", style={"padding":"16px 24px 0"}),

        html.Hr(style={"borderColor":"#dee2e6","margin":"0 24px 0"}),

        # Contenido de resultados
        dcc.Loading(id="loading-main", type="dot", color="#E8A838",
                    children=html.Div(id="tab-content",
                                      style={"padding":"16px 24px 40px"})),
    ], style={"marginLeft":"200px","minHeight":"100vh","background":"#f0f2f5"}),
], style={"fontFamily":"IBM Plex Mono, monospace","color":"#212529"})


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Navegación ───────────────────────────────────────────────────────────────
@app.callback(
    Output("store-active-tab","data"),
    Output("header-tab-name","children"),
    Output("config-panel","children"),
    [Input(f"btn-nav-{t}","n_clicks") for _,_,t in NAV_ITEMS],
    prevent_initial_call=False,
)
def cambiar_tab(*_):
    ctx = callback_context
    if not ctx.triggered or ctx.triggered[0]["prop_id"] == ".":
        # Estado inicial
        return "tab-demanda", "DEMANDA", CTRL_POR_TAB["tab-demanda"]
    tab_id = ctx.triggered[0]["prop_id"].split(".")[0].replace("btn-nav-","")
    label  = next(l for _,l,t in NAV_ITEMS if t == tab_id)
    return tab_id, label, CTRL_POR_TAB.get(tab_id, html.Div())


# ── Mostrar capacidad laboral en tiempo real ──────────────────────────────────
@app.callback(
    Output("agr-cap-display","children"),
    Input("agr-trab","value"), Input("agr-hsem","value"),
    Input("agr-nsem","value"), Input("agr-ef","value"),
    Input("agr-aus","value"),
    prevent_initial_call=True,
)
def mostrar_cap(trab, hsem, nsem, ef, aus):
    trab = trab or 10; hsem = hsem or 44
    nsem = nsem or 4;  ef   = ef   or 95; aus  = aus  or 5
    cap  = trab * hsem * nsem * ((ef - aus) / 100)
    return f"💪 Capacidad laboral efectiva: {cap:,.1f} horas-hombre / período"


# ── Planeación Agregada ───────────────────────────────────────────────────────
@app.callback(
    Output("store-agr","data"),
    Output("agr-status","children"),
    Input("btn-agr","n_clicks"),
    State("agr-ct","value"),  State("agr-ht","value"),   State("agr-pit","value"),
    State("agr-invseg","value"),
    State("agr-crt","value"), State("agr-cot","value"),
    State("agr-cwmas","value"), State("agr-cwmenos","value"),
    State("agr-trab","value"), State("agr-hsem","value"),
    State("agr-nsem","value"), State("agr-maxdw","value"),
    State("agr-m","value"),   State("agr-ef","value"),   State("agr-aus","value"),
    prevent_initial_call=True,
)
def calcular_agregacion(n,
    ct, ht, pit, invseg,
    crt, cot, cwm, cwd,
    trab, hsem, nsem, maxdw,
    M, ef, aus):
    if not n: return dash.no_update, dash.no_update
    try:
        trab=trab or 10; hsem=hsem or 44; nsem=nsem or 4
        ef=ef or 95;     aus=aus or 5;    M=M or 1
        lr = trab * hsem * nsem * ((ef - aus) / 100)

        params = {
            "Ct": ct or 4310, "Ht": ht or 100000, "PIt": pit or 100000,
            "CRt": crt or 11364, "COt": cot or 14205,
            "CW_mas": cwm or 14204, "CW_menos": cwd or 15061,
            "M": M, "LR_inicial": lr,
            "delta_trabajadores": maxdw or 50,
        }
        df, costo = run_agregacion(params)
        status = f"✓ Óptimo — Costo total: COP ${costo:,.0f} | Cap. laboral: {lr:,.0f} H-H"
        return df.to_json(), status
    except Exception as e:
        return None, f"✗ Error: {e}"


# ── Desagregación ─────────────────────────────────────────────────────────────
@app.callback(
    Output("store-desag","data"),
    Output("desag-status","children"),
    Input("btn-desag","n_clicks"),
    State("store-agr","data"),
    State("desag-cprod","value"), State("desag-cinv","value"),
    prevent_initial_call=True,
)
def calcular_desagregacion(n, agr_j, cprod, cinv):
    if not n: return dash.no_update, dash.no_update
    if not agr_j:
        return None, "⚠ Calcula el plan agregado primero."
    try:
        df_agr   = pd.read_json(agr_j)
        prod_hh  = dict(zip(df_agr["Mes"], df_agr["Produccion_HH"]))
        desag    = run_desagregacion(prod_hh, cprod or 1.0, cinv or 1.0)
        status   = f"✓ Desagregación completada — {len(PRODUCTOS)} productos, {len(MESES)} meses"
        return {p: df.to_json() for p, df in desag.items()}, status
    except Exception as e:
        return None, f"✗ Error: {e}"


# ── Simulación ────────────────────────────────────────────────────────────────
@app.callback(
    Output("store-sim","data"),
    Output("store-util","data"),
    Output("store-kpis-base","data"),
    Output("store-sens","data"),
    Output("store-plan-mes","data"),
    Output("sim-status","children"),
    Input("btn-sim","n_clicks"),
    State("store-desag","data"),
    State("sim-mes","value"),      State("sim-ft","value"),
    State("sim-opciones","value"), State("sim-semilla","value"),
    State("sim-nmezcla","value"),  State("sim-ndosif","value"),
    State("sim-nhorno","value"),   State("sim-nenfr","value"),
    State("sim-nempaq","value"),   State("sim-namasad","value"),
    State("sim-t-mezcla","value"), State("sim-t-dosif","value"),
    State("sim-t-horno","value"),  State("sim-t-enfr","value"),
    State("sim-t-empaq","value"),  State("sim-t-amasad","value"),
    prevent_initial_call=True,
)
def ejecutar_simulacion(n, desag_j, mes_idx, ft, opciones, semilla,
                        nm, nd, nh, ne, nep, na,
                        tm, td, th, te, tep, ta):
    if not n: return [dash.no_update]*6
    if not desag_j:
        return None,None,None,None,None,"⚠ Calcula la desagregación primero."
    try:
        desag   = {p: pd.read_json(v) for p, v in desag_j.items()}
        mes_nm  = MESES[mes_idx or 1]
        plan_mes = {
            p: int(desag[p].loc[desag[p]["Mes"]==mes_nm, "Produccion"].values[0])
            for p in PRODUCTOS
        }

        cap_r = {"mezcla":nm or 2,"dosificado":nd or 2,"horno":nh or 3,
                 "enfriamiento":ne or 4,"empaque":nep or 2,"amasado":na or 1}
        tiempos = {"mezcla":tm or [12,18],"dosificado":td or [8,24],
                   "horno":th or [20,48],"enfriamiento":te or [25,72],
                   "empaque":tep or [4,12],"amasado":ta or [16,24]}
        factor_t = ft or 1.0
        if "turno" in (opciones or []): factor_t = 0.80
        falla    = "falla" in (opciones or [])

        df_l, df_u, df_s = run_simulacion(
            plan_mes, cap_r, falla, factor_t,
            TAMANO_LOTE_BASE.copy(), semilla or 42, tiempos,
        )
        df_kpi = calc_kpis(df_l, plan_mes)
        df_ut  = calc_utilizacion(df_u)

        status = f"✓ {len(df_l)} lotes procesados — mes: {mes_nm}"
        return (
            df_l.to_json()   if not df_l.empty  else "{}",
            df_u.to_json()   if not df_u.empty  else "{}",
            df_kpi.to_json() if not df_kpi.empty else "{}",
            df_s.to_json()   if not df_s.empty  else "{}",
            plan_mes,
            status,
        )
    except Exception as e:
        return None,None,None,None,None, f"✗ Error: {e}"


# ── Escenarios ────────────────────────────────────────────────────────────────
@app.callback(
    Output("store-esc","data"),
    Output("esc-status","children"),
    Input("btn-esc","n_clicks"),
    State("esc-sels","value"),
    State("store-plan-mes","data"),
    State("store-esc","data"),
    prevent_initial_call=True,
)
def correr_escenarios(n, sels, plan_mes, esc_store):
    if not n: return dash.no_update, dash.no_update
    if not plan_mes:
        return esc_store or {}, "⚠ Ejecuta la simulación base primero."
    if not sels:
        return esc_store or {}, "⚠ Selecciona al menos un escenario."

    ESC_DEF = {
        "base":        {"fd":1.0,"falla":False,"ft":1.0,"dh":0,"fl":1.0},
        "demanda_20":  {"fd":1.2,"falla":False,"ft":1.0,"dh":0,"fl":1.0},
        "falla_horno": {"fd":1.0,"falla":True, "ft":1.0,"dh":0,"fl":1.0},
        "red_cap":     {"fd":1.0,"falla":False,"ft":1.0,"dh":-1,"fl":1.0},
        "doble_turno": {"fd":1.0,"falla":False,"ft":0.80,"dh":0,"fl":1.0},
        "lote_grande": {"fd":1.0,"falla":False,"ft":1.0,"dh":0,"fl":1.5},
        "optimizado":  {"fd":1.0,"falla":False,"ft":0.85,"dh":1,"fl":1.0},
    }
    resultado = dict(esc_store or {})
    try:
        for nm in sels:
            cfg     = ESC_DEF.get(nm, ESC_DEF["base"])
            plan_aj = {p: max(int(u*cfg["fd"]), 0) for p, u in plan_mes.items()}
            cap_r   = {**CAPACIDAD_BASE,
                       "horno": max(CAPACIDAD_BASE["horno"] + cfg.get("dh",0), 1)}
            tam_l   = {p: max(int(t*cfg.get("fl",1.0)), 1)
                       for p, t in TAMANO_LOTE_BASE.items()}
            df_l, df_u, _ = run_simulacion(plan_aj, cap_r, cfg["falla"], cfg["ft"], tam_l)
            dk = calc_kpis(df_l, plan_aj)
            du = calc_utilizacion(df_u)
            resultado[nm] = {
                "kpis": dk.to_json() if not dk.empty else "{}",
                "util": du.to_json() if not du.empty else "{}",
            }
        status = f"✓ {len(sels)} escenarios calculados — ver pestaña KPIs"
        return resultado, status
    except Exception as e:
        return resultado, f"✗ Error: {e}"


# ── Renderizar contenido del tab ──────────────────────────────────────────────
@app.callback(
    Output("tab-content","children"),
    Input("store-active-tab","data"),
    State("store-agr","data"),    State("store-desag","data"),
    State("store-sim","data"),    State("store-util","data"),
    State("store-kpis-base","data"), State("store-sens","data"),
    State("store-plan-mes","data"),  State("store-esc","data"),
)
def render_tab(tab, agr_j, desag_j, sim_j, util_j, kpi_j, sen_j, plan_mes, esc_store):

    # ── DEMANDA ───────────────────────────────────────────────────────────────
    if tab == "tab-demanda":
        total = sum(sum(DEM_HISTORICA[p]) for p in PRODUCTOS)
        pico  = max(MESES, key=lambda m: sum(DEM_HISTORICA[p][MESES.index(m)] for p in PRODUCTOS))
        valle = min(MESES, key=lambda m: sum(DEM_HISTORICA[p][MESES.index(m)] for p in PRODUCTOS))
        total_anual_und = sum(sum(DEM_HISTORICA[p]) for p in PRODUCTOS)
        return html.Div([
            sec_titulo("DEMANDA HISTÓRICA", "datos fijos — registros internos Dora del Hoyo"),
            dbc.Row([
                kpi_card("Total Anual",  f"{total_anual_und:,}", "und",  "#E8A838"),
                kpi_card("Productos",    len(PRODUCTOS),          "",     "#4FC3F7"),
                kpi_card("Mes Pico",     pico,                   "",     "#81C784"),
                kpi_card("Mes Valle",    valle,                  "",     "#CE93D8"),
                kpi_card("Meses",        len(MESES),             "",     "#FF8A65"),
            ], className="g-3 mb-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_demanda_barras(),
                                  config={"displayModeBar":False}), width=8),
                dbc.Col(dcc.Graph(figure=fig_demanda_heatmap(),
                                  config={"displayModeBar":False}), width=4),
            ], className="g-3 mb-4"),
            dcc.Graph(figure=fig_demanda_lineas(), config={"displayModeBar":False}),
            html.Div(style={"height":"16px"}),
            sec_titulo("TABLA DE DEMANDA HISTÓRICA", "unidades por mes por producto"),
            tabla_dash(
                pd.DataFrame(DEM_HISTORICA, index=MESES)
                  .reset_index().rename(columns={"index":"Mes"}),
                "tbl-dem",
            ),
        ])

    # ── PLANEACIÓN AGREGADA ───────────────────────────────────────────────────
    if tab == "tab-agregacion":
        if not agr_j:
            return _no_data("Configura los parámetros arriba y presiona ▶ CALCULAR PLAN AGREGADO")
        df_agr = pd.read_json(agr_j)
        costo  = df_agr["Produccion_HH"].sum() * 4310
        return html.Div([
            sec_titulo("PLAN AGREGADO","optimización PuLP/CBC — demanda fija, parámetros configurables"),
            dbc.Row([
                kpi_card("Prod. Total HH", f"{df_agr['Produccion_HH'].sum():,.0f}", "H-H", "#E8A838"),
                kpi_card("Backlog Total",  f"{df_agr['Backlog_HH'].sum():,.1f}",    "H-H", "#FF8A65"),
                kpi_card("H. Extra Total", f"{df_agr['Horas_Extras'].sum():,.1f}",  "H-H", "#4FC3F7"),
                kpi_card("Contrataciones", f"{df_agr['Contratacion'].sum():,.0f}",  "",    "#81C784"),
                kpi_card("Despidos",       f"{df_agr['Despidos'].sum():,.0f}",      "",    "#CE93D8"),
            ], className="g-3 mb-4"),
            dcc.Graph(figure=fig_agregacion(df_agr, costo), config={"displayModeBar":False}),
            html.Div(style={"height":"16px"}),
            dcc.Graph(figure=fig_fuerza_laboral(df_agr), config={"displayModeBar":False}),
            html.Div(style={"height":"16px"}),
            sec_titulo("TABLA DETALLADA", "plan mensual completo en H-H"),
            tabla_dash(df_agr, "tbl-agr"),
        ])

    # ── DESAGREGACIÓN ─────────────────────────────────────────────────────────
    if tab == "tab-desag":
        if not desag_j:
            return _no_data("Configura los costos arriba y presiona ▶ CALCULAR DESAGREGACIÓN")
        desag_dict = {p: pd.read_json(v) for p, v in desag_j.items()}
        total_und  = sum(desag_dict[p]["Produccion"].sum() for p in PRODUCTOS)
        return html.Div([
            sec_titulo("DESAGREGACIÓN DEL PLAN","unidades por producto y mes"),
            dbc.Row([
                kpi_card("Total Unidades", f"{total_und:,.0f}", "und", "#E8A838"),
                *[kpi_card(
                    p.replace("_"," ")[:12],
                    f"{desag_dict[p]['Produccion'].sum():,.0f}", "und",
                    PROD_COLORS[p],
                ) for p in PRODUCTOS],
            ], className="g-3 mb-4"),
            dcc.Graph(figure=fig_desagregacion_grid(desag_dict), config={"displayModeBar":False}),
            html.Div(style={"height":"20px"}),
            sec_titulo("INVENTARIOS Y BACKLOG POR PRODUCTO"),
            *[html.Div([
                html.Div(p.replace("_"," "),
                         style={"fontFamily":"Barlow Condensed, sans-serif","fontWeight":"700",
                                "color":PROD_COLORS[p],"marginBottom":"6px","fontSize":"16px"}),
                tabla_dash(desag_dict[p], f"tbl-desag-{p}", page_size=6),
                html.Div(style={"height":"12px"}),
            ]) for p in PRODUCTOS],
        ])

    # ── SIMULACIÓN ────────────────────────────────────────────────────────────
    if tab == "tab-sim":
        if not sim_j or sim_j == "{}":
            return _no_data("Configura los parámetros arriba y presiona ▶ EJECUTAR SIMULACIÓN")
        df_l  = pd.read_json(sim_j)
        df_u  = pd.read_json(util_j) if util_j and util_j != "{}" else pd.DataFrame()
        n_lotes  = len(df_l)
        dur_prom = round(df_l["tiempo_sistema"].mean(), 1) if not df_l.empty else 0
        return html.Div([
            sec_titulo("SIMULACIÓN DE EVENTOS DISCRETOS","SimPy — flujo de lotes por estación"),
            dbc.Row([
                kpi_card("Lotes Simulados",  n_lotes,           "",    "#E8A838"),
                kpi_card("Tiempo Prom Lote", f"{dur_prom:,.1f}","min", "#4FC3F7"),
                kpi_card("Productos",
                         len(df_l["producto"].unique()) if not df_l.empty else 0,
                         "", "#81C784"),
                kpi_card("Unidades Totales",
                         f"{df_l['tamano'].sum():,}" if not df_l.empty else "0",
                         "und", "#CE93D8"),
            ], className="g-3 mb-4"),
            dcc.Graph(figure=fig_gantt(df_l, 80), config={"displayModeBar":False}),
            html.Div(style={"height":"16px"}),
            dcc.Graph(figure=fig_colas(df_u), config={"displayModeBar":False}),
            html.Div(style={"height":"16px"}),
            sec_titulo("REGISTRO DE LOTES"),
            tabla_dash(
                df_l[["lote_id","producto","tamano","t_creacion","t_fin",
                       "tiempo_sistema","total_espera"]].head(200),
                "tbl-lotes",
            ),
        ])

    # ── KPIs ──────────────────────────────────────────────────────────────────
    if tab == "tab-kpis":
        # Parte 1: KPIs base (simulación inicial)
        kpi_base_section = _no_data("Ejecuta la simulación base (pestaña SIMULACIÓN) para ver los KPIs iniciales.")
        if kpi_j and kpi_j != "{}":
            df_kpi = pd.read_json(kpi_j)
            df_ut  = pd.read_json(util_j) if util_j and util_j != "{}" else pd.DataFrame()
            gauges = fig_utilizacion_gauge(df_ut)
            kpi_base_section = html.Div([
                sec_titulo("KPIs BASE — SIMULACIÓN INICIAL",
                           "métricas del escenario base configurado en pestaña Simulación"),
                # Gauges de utilización
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=g, config={"displayModeBar":False}), width=2)
                    for g in gauges
                ], className="g-2 mb-4") if gauges else html.Div(),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_kpis_radar(df_kpi),
                                      config={"displayModeBar":False}), width=6),
                    dbc.Col([
                        sec_titulo("TABLA KPIs POR PRODUCTO"),
                        tabla_dash(df_kpi, "tbl-kpis-base", page_size=6),
                    ], width=6),
                ], className="g-3 mb-4"),
                tabla_dash(df_ut, "tbl-util-base", page_size=8) if not df_ut.empty else html.Div(),
            ])

        # Parte 2: KPIs de escenarios
        esc_section = html.Div()
        if esc_store:
            res_p = {
                nm: {
                    "kpis": pd.read_json(v["kpis"]) if v.get("kpis","{}") != "{}" else pd.DataFrame(),
                    "util": pd.read_json(v["util"]) if v.get("util","{}") != "{}" else pd.DataFrame(),
                }
                for nm, v in esc_store.items()
            }
            filas_esc = []
            for nm, res in res_p.items():
                dk = res.get("kpis", pd.DataFrame())
                du = res.get("util", pd.DataFrame())
                if dk.empty: continue
                fila = {"Escenario": nm}
                for col in ["Throughput (und/h)","Lead Time (min/lote)","WIP Prom","Cumplimiento %"]:
                    if col in dk.columns: fila[col] = round(dk[col].mean(), 2)
                if not du.empty and "Utilización_%" in du.columns:
                    fila["Util Máx %"] = round(du["Utilización_%"].max(), 2)
                filas_esc.append(fila)
            df_esc_tabla = pd.DataFrame(filas_esc)

            esc_section = html.Div([
                html.Hr(style={"borderColor":"#dee2e6","margin":"32px 0"}),
                sec_titulo("KPIs POR ESCENARIO","comparación de escenarios what-if calculados"),
                dcc.Graph(figure=fig_comparacion_escenarios(res_p),
                          config={"displayModeBar":False}),
                html.Div(style={"height":"16px"}),
                sec_titulo("TABLA COMPARATIVA DE ESCENARIOS"),
                tabla_dash(df_esc_tabla, "tbl-kpis-esc", page_size=10),
            ])

        return html.Div([kpi_base_section, esc_section])

    # ── SENSORES ──────────────────────────────────────────────────────────────
    if tab == "tab-sensores":
        if not sen_j or sen_j == "{}":
            return _no_data("Ejecuta la simulación base para generar datos de sensores.")
        df_s   = pd.read_json(sen_j)
        t_max  = round(df_s["temperatura"].max(), 1) if not df_s.empty else 0
        t_min  = round(df_s["temperatura"].min(), 1) if not df_s.empty else 0
        t_prom = round(df_s["temperatura"].mean(),1) if not df_s.empty else 0
        alarmas = int((df_s["temperatura"] > 200).sum()) if not df_s.empty else 0
        return html.Div([
            sec_titulo("SENSORES VIRTUALES","monitoreo en tiempo real — horno de producción"),
            dbc.Row([
                kpi_card("Temp. Máxima",   t_max,      "°C", "#FF8A65"),
                kpi_card("Temp. Mínima",   t_min,      "°C", "#4FC3F7"),
                kpi_card("Temp. Promedio", t_prom,     "°C", "#E8A838"),
                kpi_card("Lecturas",       len(df_s),  "",   "#81C784"),
                kpi_card("Alarmas >200°C", alarmas,    "",   "#F06292"),
            ], className="g-3 mb-4"),
            dcc.Graph(figure=fig_sensores(df_s), config={"displayModeBar":False}),
            html.Div(style={"height":"16px"}),
            sec_titulo("REGISTRO DE LECTURAS"),
            tabla_dash(df_s.round(2), "tbl-sensores", page_size=12),
        ])

    # ── ESCENARIOS ────────────────────────────────────────────────────────────
    if tab == "tab-escenarios":
        has_esc = bool(esc_store)
        return html.Div([
            sec_titulo("ANÁLISIS DE ESCENARIOS WHAT-IF",
                       "selecciona escenarios arriba y presiona ▶ CORRER ESCENARIOS"),
            html.Div([
                html.Div("📋 Descripción de escenarios disponibles",
                         style={**LABEL_S,"fontSize":"12px","color":"#E8A838",
                                "marginBottom":"12px","textTransform":"none"}),
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Escenario",  style={"fontFamily":"IBM Plex Mono, monospace","fontSize":"11px"}),
                        html.Th("Parámetro modificado",  style={"fontFamily":"IBM Plex Mono, monospace","fontSize":"11px"}),
                        html.Th("Valor",      style={"fontFamily":"IBM Plex Mono, monospace","fontSize":"11px"}),
                        html.Th("Objetivo",   style={"fontFamily":"IBM Plex Mono, monospace","fontSize":"11px"}),
                    ])),
                    html.Tbody([
                        html.Tr([html.Td(c) for c in row],
                                style={"fontFamily":"IBM Plex Mono, monospace","fontSize":"11px"})
                        for row in [
                            ["Base",         "—",                         "—",         "Referencia sin cambios"],
                            ["Demanda +20%", "Factor demanda",            "× 1.2",     "Impacto de crecimiento de ventas"],
                            ["Falla horno",  "Tiempo adicional en horno", "+10–30 min","Impacto de falla de equipo"],
                            ["Reducir cap.", "N° hornos",                 "−1",        "Efecto de reducción de capacidad"],
                            ["Doble turno",  "Factor tiempo proceso",     "× 0.80",    "Mejora por segundo turno"],
                            ["Lotes +50%",   "Tamaño de lote",            "× 1.5",     "Economías de escala"],
                            ["Optimizado",   "+1 horno + tiempos −15%",   "mixto",     "Escenario de mejor práctica"],
                        ]
                    ]),
                ], bordered=True, hover=True, size="sm",
                   style={"marginBottom":"0"}),
            ], style={**CARD_S,"marginBottom":"24px"}),

            html.Div(
                _no_data("Selecciona escenarios arriba y presiona ▶ CORRER ESCENARIOS")
                if not has_esc else html.Div([
                    sec_titulo("RESULTADOS DE ESCENARIOS",
                               f"{len(esc_store)} escenarios calculados — ver comparación completa en pestaña KPIs"),
                    html.Div([
                        html.Div(
                            f"✓ {nm}",
                            style={"display":"inline-block","background":"#d4edda",
                                   "border":"1px solid #c3e6cb","borderRadius":"4px",
                                   "padding":"4px 12px","marginRight":"8px","marginBottom":"8px",
                                   "fontSize":"12px","fontFamily":"IBM Plex Mono, monospace",
                                   "color":"#155724"},
                        )
                        for nm in esc_store.keys()
                    ], style={"marginBottom":"16px"}),
                    html.Div("→ Ve a la pestaña KPIs para ver la comparación completa y los gráficos.",
                             style={"color":"#4FC3F7","fontSize":"12px",
                                    "fontFamily":"IBM Plex Mono, monospace"}),
                ])
            ),
        ])

    return html.Div("Selecciona una sección.",
                    style={"color":"#6c757d","padding":"40px"})


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  GEMELO DIGITAL — DORA DEL HOYO  v3.0")
    print("  http://127.0.0.1:8050")
    print("═"*60 + "\n")
    app.run(debug=True, port=8050)
