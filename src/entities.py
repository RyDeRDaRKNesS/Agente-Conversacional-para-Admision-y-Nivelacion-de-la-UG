"""
Extracción de entidades del dominio (RF-05).

Entidades soportadas:
- FECHA: fechas en formatos comunes (dd/mm/yyyy, dd-mm-yyyy, "10 de julio", etc.)
- CEDULA: números de cédula ecuatoriana (10 dígitos).
- CARRERA: nombres de carreras conocidas de la UG.
- TERMINO_DOMINIO: expresiones fijas del proceso de admisión/nivelación.
"""

import re

MESES = (
    "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|"
    "octubre|noviembre|diciembre"
)

PATRON_FECHA = re.compile(
    rf"\b(\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|\d{{1,2}}\s+de\s+(?:{MESES})(?:\s+de\s+\d{{4}})?)\b",
    re.IGNORECASE,
)

PATRON_CEDULA = re.compile(r"\b\d{10}\b")

CARRERAS_CONOCIDAS = [
    "ciencia de datos", "ciencia de datos e inteligencia artificial",
    "inteligencia artificial", "ingenieria en sistemas", "medicina",
    "derecho", "psicologia", "arquitectura", "ingenieria civil",
    "comunicacion social", "administracion", "contabilidad",
    "enfermeria", "odontologia",
]

TERMINOS_DOMINIO = [
    "curso de nivelacion", "cupo aceptado", "cronograma de registro",
    "creacion de cuenta", "portal de admision", "matricula ordinaria",
    "aprobacion de asignaturas", "control de asistencias",
]


def _normalizar(texto: str) -> str:
    return texto.lower()


def extraer_entidades(texto: str) -> dict:
    """
    Devuelve un diccionario con listas de entidades detectadas en el texto original
    (sin preprocesar, para no perder dígitos/formato de fechas).
    """
    texto_norm = _normalizar(texto)

    fechas = PATRON_FECHA.findall(texto)
    cedulas = PATRON_CEDULA.findall(texto)
    carreras = [c for c in CARRERAS_CONOCIDAS if c in texto_norm]
    terminos = [t for t in TERMINOS_DOMINIO if t in texto_norm]

    return {
        "FECHA": fechas,
        "CEDULA": cedulas,
        "CARRERA": carreras,
        "TERMINO_DOMINIO": terminos,
    }


if __name__ == "__main__":
    pruebas = [
        "Mi cedula es 0912345678 y quiero saber si tengo cupo aceptado",
        "El curso de nivelacion empieza el 10 de julio de 2026",
        "Quiero postular a ciencia de datos, la matricula ordinaria es el 15/07/2026",
    ]
    for p in pruebas:
        print(p)
        print(" ->", extraer_entidades(p))
        print()
