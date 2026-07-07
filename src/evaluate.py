"""
evaluate.py
============

Script de evaluación del agente (RF-08).

Responsabilidades:
    1. Ejecutar el motor de NLP sobre el set de prueba
       (data/test_queries.csv, 45 consultas etiquetadas manualmente).
    2. Reportar accuracy y F1-macro, más un reporte por clase
       (precision/recall/f1 de cada intención).
    3. Barrer distintos valores de umbral de confianza para justificar,
       con datos, el valor final elegido en `nlp_engine.UMBRAL_CONFIANZA`
       (pregunta "d" de la sección de Justificación del enunciado).
    4. Persistir los resultados en `resultados_evaluacion.txt`, para poder
       adjuntarlos como evidencia en el informe sin tener que volver a
       ejecutar el script.

Ejecución:
    python evaluate.py
"""

import csv
import os
import sys
from io import StringIO

from sklearn.metrics import accuracy_score, classification_report, f1_score

from nlp_engine import MotorNLP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PATH = os.path.join(BASE_DIR, "data", "test_queries.csv")
RESULTADOS_PATH = os.path.join(BASE_DIR, "resultados_evaluacion.txt")

# Valores de umbral candidatos para el barrido de la Sección "Justificación".
# 0.28 es el valor que finalmente se fijó en nlp_engine.UMBRAL_CONFIANZA;
# el resto se incluye únicamente para poder comparar y documentar la
# decisión con evidencia empírica.
UMBRALES_A_PROBAR = [0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50]


def cargar_casos_prueba(path: str) -> list:
    """
    Lee el CSV de consultas etiquetadas.

    Args:
        path: ruta al archivo CSV con columnas 'consulta' e
            'intent_esperado'.

    Returns:
        Lista de tuplas (consulta, intent_esperado), en el mismo orden
        del archivo.
    """
    casos = []
    with open(path, "r", encoding="utf-8") as archivo_csv:
        lector = csv.DictReader(archivo_csv)
        for fila in lector:
            casos.append((fila["consulta"], fila["intent_esperado"]))
    return casos


def evaluar_con_umbral(motor: MotorNLP, casos: list, umbral: float) -> dict:
    """
    Corre el set de prueba completo con un umbral de confianza específico,
    sin necesidad de reconstruir los vectorizadores TF-IDF (que son
    independientes del umbral): solo se sobreescribe `motor.umbral` antes
    de cada consulta.

    Args:
        motor: instancia ya inicializada de MotorNLP (se reutiliza para
            no pagar el costo de reconstruir TF-IDF en cada iteración del
            barrido de umbrales).
        casos: lista de (consulta, intent_esperado).
        umbral: valor de umbral de confianza a evaluar.

    Returns:
        Diccionario con 'accuracy', 'f1_macro' y las listas 'y_true'/'y_pred'
        usadas para calcularlos (útiles para el reporte detallado).
    """
    motor.umbral = umbral

    y_true, y_pred = [], []
    for consulta, esperado in casos:
        resultado = motor.detectar_intencion(consulta)
        y_true.append(esperado)
        y_pred.append(resultado["intent"])

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def imprimir_y_capturar(*args, buffer: StringIO, **kwargs) -> None:
    """
    Utilidad para escribir simultáneamente a stdout (para que el desarrollador
    vea el progreso en consola) y a un buffer en memoria (para poder guardar
    todo el reporte en `resultados_evaluacion.txt` al final).
    """
    print(*args, **kwargs)
    print(*args, **kwargs, file=buffer)


def evaluar() -> None:
    """
    Punto de entrada principal del script. Ejecuta:
        1. Evaluación detallada con el umbral de producción
           (nlp_engine.UMBRAL_CONFIANZA), consulta por consulta.
        2. Barrido de umbrales candidatos, para justificar la elección.
    Todo el reporte se imprime en consola y se guarda en
    `resultados_evaluacion.txt` en la raíz del repositorio.
    """
    buffer = StringIO()
    motor = MotorNLP()  # construye los vectorizadores TF-IDF una sola vez
    casos = cargar_casos_prueba(TEST_PATH)
    umbral_produccion = motor.umbral

    # ------------------------------------------------------------------
    # 1) Evaluación detallada con el umbral de producción
    # ------------------------------------------------------------------
    imprimir_y_capturar(
        f"\n=== Evaluación detallada (umbral de producción = {umbral_produccion}) ===\n",
        buffer=buffer,
    )
    imprimir_y_capturar(
        f"{'Consulta':42} {'Esperado':30} {'Detectado':30} {'Conf.':6}",
        buffer=buffer,
    )
    imprimir_y_capturar("-" * 112, buffer=buffer)

    y_true, y_pred = [], []
    for consulta, esperado in casos:
        resultado = motor.detectar_intencion(consulta)
        predicho = resultado["intent"]
        y_true.append(esperado)
        y_pred.append(predicho)

        marca = "OK " if predicho == esperado else "ERR"
        imprimir_y_capturar(
            f"[{marca}] {consulta[:38]:38} {esperado:30} {predicho:30} {resultado['confianza']:.3f}",
            buffer=buffer,
        )

    accuracy_final = accuracy_score(y_true, y_pred)
    f1_macro_final = f1_score(y_true, y_pred, average="macro", zero_division=0)

    imprimir_y_capturar("\n" + "=" * 50, buffer=buffer)
    imprimir_y_capturar(f"Accuracy (umbral {umbral_produccion}) : {accuracy_final:.3f}", buffer=buffer)
    imprimir_y_capturar(f"F1-macro (umbral {umbral_produccion}) : {f1_macro_final:.3f}", buffer=buffer)
    imprimir_y_capturar("=" * 50, buffer=buffer)

    imprimir_y_capturar("\nReporte por clase (umbral de producción):\n", buffer=buffer)
    imprimir_y_capturar(classification_report(y_true, y_pred, zero_division=0), buffer=buffer)

    # ------------------------------------------------------------------
    # 2) Barrido de umbrales, para justificar la elección documentada en
    #    nlp_engine.py (pregunta "d" de la sección de Justificación).
    # ------------------------------------------------------------------
    imprimir_y_capturar("=== Barrido de umbrales de confianza ===\n", buffer=buffer)
    imprimir_y_capturar(f"{'Umbral':8} {'Accuracy':10} {'F1-macro':10}", buffer=buffer)
    imprimir_y_capturar("-" * 30, buffer=buffer)

    for umbral_candidato in UMBRALES_A_PROBAR:
        resultado_barrido = evaluar_con_umbral(motor, casos, umbral_candidato)
        marca_actual = "  <- valor usado en producción" if umbral_candidato == umbral_produccion else ""
        imprimir_y_capturar(
            f"{umbral_candidato:<8.2f} {resultado_barrido['accuracy']:<10.3f} "
            f"{resultado_barrido['f1_macro']:<10.3f}{marca_actual}",
            buffer=buffer,
        )

    # Se restaura el umbral de producción por si el objeto `motor` se
    # reutiliza después de llamar a esta función.
    motor.umbral = umbral_produccion

    with open(RESULTADOS_PATH, "w", encoding="utf-8") as archivo_resultados:
        archivo_resultados.write(buffer.getvalue())

    print(f"\nResultados guardados en: {RESULTADOS_PATH}")


if __name__ == "__main__":
    evaluar()
