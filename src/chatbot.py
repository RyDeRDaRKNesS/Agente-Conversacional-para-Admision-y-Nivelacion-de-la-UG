"""
chatbot.py
===========

Interfaz de ejecución local por consola (RF-07, opción 1).

Este módulo es intencionalmente delgado: toda la lógica de negocio
(preprocesamiento, TF-IDF, similitud coseno, entidades, fallback) vive en
`nlp_engine.py`. Aquí solo se implementa el bucle de entrada/salida por
terminal, para que la misma lógica pueda reutilizarse sin duplicación desde
la interfaz web (`app.py`).

Ejecución:
    python chatbot.py
"""

from nlp_engine import MotorNLP

COMANDOS_SALIDA = {"salir", "exit", "quit"}


def main() -> None:
    """
    Bucle principal de la interfaz de consola.

    Instancia el motor NLP una sola vez (el costo de construir los
    vectorizadores TF-IDF se paga solo al arrancar, no en cada turno de
    conversación) y luego entra en un ciclo de lectura-proceso-impresión
    hasta que el usuario escriba un comando de salida o interrumpa el
    proceso (Ctrl+C / EOF).
    """
    print("=" * 60)
    print(" Asistente Virtual de Admisión y Nivelación - UG")
    print(" Escribe 'salir' para terminar.")
    print("=" * 60)

    motor = MotorNLP()

    while True:
        try:
            consulta = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Salida silenciosa y controlada ante Ctrl+D/Ctrl+C, en vez de
            # dejar que la excepción se propague como un traceback crudo.
            print("\n¡Hasta luego!")
            break

        if consulta.lower() in COMANDOS_SALIDA:
            print("Bot: ¡Hasta luego! Éxitos en tu proceso de admisión.")
            break

        # `responder()` ya maneja internamente entradas vacías, texto sin
        # sentido o errores inesperados (ver try/except en nlp_engine.py),
        # por lo que esta interfaz no necesita lógica adicional de manejo
        # de errores: siempre recibe un diccionario bien formado.
        resultado = motor.responder(consulta)

        print(f"Bot: {resultado['respuesta']}")
        print(f"     [intent={resultado['intent']} | confianza={resultado['confianza']}]")

        # Solo se muestran las entidades si el detector encontró al menos
        # una, para no ensuciar la salida en turnos donde no aplica.
        if any(resultado["entidades"].values()):
            print(f"     [entidades detectadas: {resultado['entidades']}]")


if __name__ == "__main__":
    main()
