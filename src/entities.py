"""
entities.py
============

Módulo de extracción de entidades del dominio (RF-05).

Este módulo NO usa modelos de NER entrenados ni librerías de deep learning:
por restricción del enunciado (Sección 4 - Alcance), la extracción se basa
exclusivamente en reglas y expresiones regulares, lo cual es válido para un
dominio cerrado y bien delimitado como el de admisión/nivelación.

Entidades soportadas
---------------------
FECHA
    Fechas en formato numérico (dd/mm/yyyy, dd-mm-yyyy) o textual
    ("10 de julio de 2026").
CEDULA
    Números de 10 dígitos que además se validan contra el algoritmo oficial
    de cédula ecuatoriana (módulo 10), para distinguir un número de cédula
    real de cualquier secuencia de 10 dígitos (ej. un número de teléfono
    mal escrito).
CARRERA
    Nombres de carreras conocidas de la UG, detectadas por coincidencia de
    subcadena sobre una lista curada.
TERMINO_DOMINIO
    Expresiones fijas propias del proceso de admisión/nivelación
    (ej. "cupo aceptado", "curso de nivelacion").

Notas de diseño
----------------
- La detección de entidades se hace sobre el texto ORIGINAL del usuario
  (no sobre el texto preprocesado/stemmizado), porque el stemming destruye
  información necesaria para reconocer fechas y números de cédula.
- Para CARRERA se ordena la lista de más específica a más genérica antes de
  buscar coincidencias, de modo que "medicina veterinaria" no se reporte
  incorrectamente como una simple coincidencia parcial de "medicina" cuando
  el término compuesto también está definido en la lista.
"""

import re
from typing import Dict, List

# ---------------------------------------------------------------------------
# Patrones de expresiones regulares
# ---------------------------------------------------------------------------

MESES = (
    "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|"
    "octubre|noviembre|diciembre"
)

# Acepta "10/07/2026", "10-07-26" y "10 de julio de 2026"
PATRON_FECHA = re.compile(
    rf"\b(\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|\d{{1,2}}\s+de\s+(?:{MESES})(?:\s+de\s+\d{{4}})?)\b",
    re.IGNORECASE,
)

# Cualquier secuencia de exactamente 10 dígitos; la validez real (dígito
# verificador, código de provincia) se confirma después con validar_cedula_ecuador().
PATRON_CEDULA = re.compile(r"\b\d{10}\b")

# ---------------------------------------------------------------------------
# Listas curadas del dominio
# ---------------------------------------------------------------------------

# Ordenadas de la más específica a la más genérica: esto evita que, por
# ejemplo, "medicina veterinaria" se resuelva únicamente como "medicina".
CARRERAS_CONOCIDAS = [
    "ciencia de datos e inteligencia artificial",
    "ciencia de datos",
    "inteligencia artificial",
    "ingenieria en sistemas",
    "ingenieria civil",
    "medicina veterinaria",
    "medicina",
    "derecho",
    "psicologia",
    "arquitectura",
    "comunicacion social",
    "administracion",
    "contabilidad",
    "enfermeria",
    "odontologia",
]

TERMINOS_DOMINIO = [
    "curso de nivelacion",
    "cupo aceptado",
    "cronograma de registro",
    "creacion de cuenta",
    "portal de admision",
    "matricula ordinaria",
    "aprobacion de asignaturas",
    "control de asistencias",
]

# Coeficientes del algoritmo módulo 10 usado por el Registro Civil del
# Ecuador para validar el dígito verificador de la cédula (posiciones 0-8).
_COEFICIENTES_CEDULA = [2, 1, 2, 1, 2, 1, 2, 1, 2]


def _normalizar(texto: str) -> str:
    """Normaliza a minúsculas para hacer matching insensible a mayúsculas."""
    return texto.lower()


def validar_cedula_ecuador(cedula: str) -> bool:
    """
    Valida un número de cédula ecuatoriana mediante el algoritmo oficial
    de módulo 10.

    Esto permite distinguir una cédula real de cualquier número de 10
    dígitos que el usuario haya escrito (por ejemplo, un número de celular
    o un código sin relación con el proceso de admisión).

    Reglas aplicadas:
    1. Debe tener exactamente 10 dígitos numéricos.
    2. Los dos primeros dígitos representan el código de provincia
       (01-24 en Ecuador continental e insular).
    3. El tercer dígito debe ser menor a 6 (identifica personas naturales).
    4. El décimo dígito es el verificador, calculado con coeficientes
       alternos [2,1,2,1,2,1,2,1,2] sobre las primeras 9 posiciones.

    Args:
        cedula: cadena de texto con el número a validar.

    Returns:
        True si la cédula es estructuralmente válida, False en caso
        contrario (incluye entradas no numéricas o de longitud distinta).
    """
    if not cedula.isdigit() or len(cedula) != 10:
        return False

    provincia = int(cedula[:2])
    if provincia < 1 or provincia > 24:
        return False

    tercer_digito = int(cedula[2])
    if tercer_digito >= 6:
        return False

    suma = 0
    for posicion in range(9):
        valor = int(cedula[posicion]) * _COEFICIENTES_CEDULA[posicion]
        if valor >= 10:
            valor -= 9
        suma += valor

    digito_verificador_esperado = (10 - (suma % 10)) % 10
    digito_verificador_real = int(cedula[9])

    return digito_verificador_esperado == digito_verificador_real


def _extraer_carreras(texto_norm: str) -> List[str]:
    """
    Busca coincidencias de carreras conocidas en el texto, evitando que una
    carrera genérica "tape" a una más específica que también aparece en el
    texto (ej. no reportar solo "medicina" si el texto dice "medicina
    veterinaria" y ambas están en CARRERAS_CONOCIDAS).
    """
    encontradas: List[str] = []
    texto_restante = texto_norm

    for carrera in CARRERAS_CONOCIDAS:  # ya viene ordenada específica -> genérica
        if carrera in texto_restante:
            encontradas.append(carrera)
            # Se "consume" la coincidencia para no volver a matchear la
            # variante genérica sobre el mismo fragmento de texto.
            texto_restante = texto_restante.replace(carrera, " ")

    return encontradas


def extraer_entidades(texto: str) -> Dict[str, List]:
    """
    Punto de entrada del módulo: extrae todas las entidades soportadas de
    una consulta en lenguaje natural.

    Args:
        texto: consulta original del usuario, SIN preprocesar (se necesita
            el texto crudo para no perder dígitos ni tildes relevantes).

    Returns:
        Diccionario con cuatro claves fijas (FECHA, CEDULA, CARRERA,
        TERMINO_DOMINIO). CEDULA devuelve una lista de diccionarios
        {"valor": ..., "valida": bool} en vez de solo strings, para que el
        consumidor (interfaz de consola o API web) pueda decidir si avisar
        al usuario que el número no parece una cédula válida.
    """
    if not texto:
        return {"FECHA": [], "CEDULA": [], "CARRERA": [], "TERMINO_DOMINIO": []}

    texto_norm = _normalizar(texto)

    fechas = PATRON_FECHA.findall(texto)
    cedulas_crudas = PATRON_CEDULA.findall(texto)
    cedulas = [
        {"valor": c, "valida": validar_cedula_ecuador(c)}
        for c in cedulas_crudas
    ]
    carreras = _extraer_carreras(texto_norm)
    terminos = [t for t in TERMINOS_DOMINIO if t in texto_norm]

    return {
        "FECHA": fechas,
        "CEDULA": cedulas,
        "CARRERA": carreras,
        "TERMINO_DOMINIO": terminos,
    }


if __name__ == "__main__":
    # Pruebas manuales rápidas del módulo (no reemplazan al set de evaluación
    # formal de src/evaluate.py, solo sirven para depuración rápida).
    pruebas = [
        "Mi cedula es 0912345678 y quiero saber si tengo cupo aceptado",
        "El curso de nivelacion empieza el 10 de julio de 2026",
        "Quiero postular a medicina veterinaria, la matricula ordinaria es el 15/07/2026",
        "mi cedula 1234567890 no es valida en realidad",
    ]
    for p in pruebas:
        print(p)
        print(" ->", extraer_entidades(p))
        print()
