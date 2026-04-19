"""
datos.py
========
Datos maestros compartidos por todos los módulos del Gemelo Digital.
"""

PRODUCTOS = ["Brownies", "Mantecadas", "Mantecadas_Amapola", "Torta_Naranja", "Pan_Maiz"]

MESES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MESES_C = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
           "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

DEM_HISTORICA = {
    "Brownies":           [315, 804, 734, 541, 494,  59, 315, 803, 734, 541, 494,  59],
    "Mantecadas":         [125, 780, 432, 910, 275,  68, 512, 834, 690, 455, 389, 120],
    "Mantecadas_Amapola": [320, 710, 520, 251, 631, 150, 330, 220, 710, 610, 489, 180],
    "Torta_Naranja":      [100, 250, 200, 101, 190,  50, 100, 220, 200, 170, 180, 187],
    "Pan_Maiz":           [330, 140, 143,  73,  83,  48,  70,  89, 118,  83,  67,  87],
}

HORAS_PRODUCTO = {
    "Brownies": 0.866, "Mantecadas": 0.175, "Mantecadas_Amapola": 0.175,
    "Torta_Naranja": 0.175, "Pan_Maiz": 0.312,
}

INV_INICIAL = {p: 0 for p in PRODUCTOS}

RUTAS = {
    "Brownies": [
        ("Mezclado",     "mezcla",       12, 18),
        ("Moldeado",     "dosificado",    8, 14),
        ("Horneado",     "horno",        30, 40),
        ("Enfriamiento", "enfriamiento", 25, 35),
        ("Corte_Empaque","empaque",       8, 12),
    ],
    "Mantecadas": [
        ("Mezclado",     "mezcla",       12, 18),
        ("Dosificado",   "dosificado",   16, 24),
        ("Horneado",     "horno",        20, 30),
        ("Enfriamiento", "enfriamiento", 35, 55),
        ("Empaque",      "empaque",       4,  6),
    ],
    "Mantecadas_Amapola": [
        ("Mezclado",     "mezcla",       12, 18),
        ("Inc_Semillas", "mezcla",        8, 12),
        ("Dosificado",   "dosificado",   16, 24),
        ("Horneado",     "horno",        20, 30),
        ("Enfriamiento", "enfriamiento", 36, 54),
        ("Empaque",      "empaque",       4,  6),
    ],
    "Torta_Naranja": [
        ("Mezclado",     "mezcla",       16, 24),
        ("Dosificado",   "dosificado",    8, 12),
        ("Horneado",     "horno",        32, 48),
        ("Enfriamiento", "enfriamiento", 48, 72),
        ("Desmolde",     "dosificado",    8, 12),
        ("Empaque",      "empaque",       8, 12),
    ],
    "Pan_Maiz": [
        ("Mezclado",     "mezcla",       12, 18),
        ("Amasado",      "amasado",      16, 24),
        ("Moldeado",     "dosificado",   12, 18),
        ("Horneado",     "horno",        28, 42),
        ("Enfriamiento", "enfriamiento", 36, 54),
        ("Empaque",      "empaque",       4,  6),
    ],
}

TAMANO_LOTE_BASE = {
    "Brownies": 12, "Mantecadas": 10, "Mantecadas_Amapola": 10,
    "Torta_Naranja": 12, "Pan_Maiz": 15,
}

CAPACIDAD_BASE = {
    "mezcla": 2, "dosificado": 2, "horno": 3,
    "enfriamiento": 4, "empaque": 2, "amasado": 1,
}

PARAMS_AGRE = {
    "Ct": 4310, "Ht": 100000, "PIt": 100000,
    "CRt": 11364, "COt": 14205,
    "CW_mas": 14204, "CW_menos": 15061,
    "M": 1, "LR_inicial": 44 * 4 * 10,
}

PROD_COLORS = {
    "Brownies":           "#6366F1",
    "Mantecadas":         "#0EA5E9",
    "Mantecadas_Amapola": "#10B981",
    "Torta_Naranja":      "#F59E0B",
    "Pan_Maiz":           "#EC4899",
}
