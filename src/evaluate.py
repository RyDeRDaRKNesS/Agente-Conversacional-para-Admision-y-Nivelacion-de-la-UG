"""
Evaluación del agente (RF-08).
Ejecutar con: python evaluate.py

Calcula accuracy y F1-macro sobre data/test_queries.csv, y muestra el reporte
por clase para poder discutir limitaciones en el informe.
"""

import csv
import os

from sklearn.metrics import accuracy_score, classification_report, f1_score

from nlp_engine import MotorNLP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PATH = os.path.join(BASE_DIR, "data", "test_queries.csv")


def cargar_casos_prueba(path: str):
    casos = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            casos.append((fila["consulta"], fila["intent_esperado"]))
    return casos


def evaluar():
    motor = MotorNLP()
    casos = cargar_casos_prueba(TEST_PATH)

    y_true, y_pred = [], []
    print(f"{'Consulta':45} {'Esperado':30} {'Detectado':30} {'Conf.':6}")
    print("-" * 115)

    for consulta, esperado in casos:
        resultado = motor.detectar_intencion(consulta)
        predicho = resultado["intent"]
        y_true.append(esperado)
        y_pred.append(predicho)

        marca = "OK " if predicho == esperado else "ERR"
        print(f"[{marca}] {consulta[:40]:40} {esperado:30} {predicho:30} {resultado['confianza']:.3f}")

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 50)
    print(f"Accuracy   : {acc:.3f}")
    print(f"F1-macro   : {f1_macro:.3f}")
    print("=" * 50)
    print("\nReporte por clase:\n")
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    evaluar()
