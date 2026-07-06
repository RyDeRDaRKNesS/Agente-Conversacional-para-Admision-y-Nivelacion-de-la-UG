"""
Interfaz de ejecución local por consola (RF-07).
Ejecutar con: python chatbot.py
"""

from nlp_engine import MotorNLP


def main():
    print("=" * 60)
    print(" Asistente Virtual de Admisión y Nivelación - UG")
    print(" Escribe 'salir' para terminar.")
    print("=" * 60)

    motor = MotorNLP()

    while True:
        try:
            consulta = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n¡Hasta luego!")
            break

        if consulta.lower() in {"salir", "exit", "quit"}:
            print("Bot: ¡Hasta luego! Éxitos en tu proceso de admisión.")
            break

        resultado = motor.responder(consulta)
        print(f"Bot: {resultado['respuesta']}")
        print(f"     [intent={resultado['intent']} | confianza={resultado['confianza']}]")
        if any(resultado["entidades"].values()):
            print(f"     [entidades detectadas: {resultado['entidades']}]")


if __name__ == "__main__":
    main()
