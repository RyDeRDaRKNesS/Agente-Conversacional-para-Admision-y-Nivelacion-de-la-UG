"""
app.py
=======

Interfaz web del agente conversacional (RF-07, opción 2) y punto de
entrada para el despliegue en producción (Render / PythonAnywhere).

Diseño de la API:
    GET  /            -> sirve la página HTML del chat (templates/index.html)
    POST /api/chat     -> recibe {"mensaje": str}, devuelve el diccionario
                          completo de nlp_engine.MotorNLP.responder()
    GET  /health        -> endpoint de salud para monitoreo del hosting

Ejecución local (servidor de desarrollo de Flask):
    python app.py

Ejecución en producción (Render, vía gunicorn):
    gunicorn wsgi:app --bind 0.0.0.0:$PORT
    (ver wsgi.py en la raíz del repositorio, que importa `app` desde aquí)
"""

import os
import sys

from flask import Flask, jsonify, render_template, request

# Se agrega el directorio actual (src/) al principio de sys.path para que
# `from nlp_engine import MotorNLP` funcione tanto si el proceso se lanza
# como `python app.py` desde dentro de src/, como si se lanza desde la raíz
# del repositorio vía `wsgi.py` (que también manipula sys.path, ver ese
# archivo). Es una medida defensiva para evitar errores de import según
# el directorio de trabajo con el que arranque cada plataforma de hosting.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nlp_engine import MotorNLP  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# El motor de NLP se instancia UNA sola vez, a nivel de módulo, cuando el
# servidor arranca (no dentro de la función `chat()`). Esto es importante
# en producción: si se reconstruyeran los vectorizadores TF-IDF en cada
# request, cada consulta del usuario pagaría el costo de reajustar TF-IDF
# sobre todo el corpus de utterances, además de introducir condiciones de
# carrera si dos requests llegan en paralelo mientras el motor se reconstruye.
motor = MotorNLP()


@app.route("/")
def index():
    """Sirve la interfaz de chat (HTML + CSS + JS embebido)."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Endpoint principal de la API conversacional.

    Espera un JSON con la forma {"mensaje": "<texto del usuario>"} y
    delega toda la lógica de PLN a `MotorNLP.responder()`.

    El manejo de errores aquí es una segunda capa de seguridad: aunque
    `nlp_engine.py` ya envuelve su propia lógica en try/except (RF: manejo
    robusto de entradas problemáticas), este endpoint también captura
    cualquier excepción no anticipada (ej. un payload JSON malformado) para
    garantizar que el servidor NUNCA responda con un error 500 crudo al
    frontend, sino con un JSON de fallback consistente con el resto de la
    API.

    Returns:
        JSON con las claves: consulta, intent, confianza, respuesta,
        entidades (y opcionalmente 'error' para depuración interna).
    """
    try:
        data = request.get_json(silent=True) or {}
        mensaje = data.get("mensaje", "")
        resultado = motor.responder(mensaje)
        return jsonify(resultado)
    except Exception as error:
        return jsonify({
            "consulta": "",
            "intent": "fallback",
            "confianza": 0.0,
            "respuesta": "Ocurrió un problema procesando tu mensaje. ¿Puedes intentar de nuevo?",
            "entidades": {},
            "error": str(error),
        }), 200


@app.route("/health")
def health():
    """
    Endpoint de verificación de salud (liveness check).

    Tanto Render como PythonAnywhere pueden usar una ruta como esta para
    confirmar que el proceso está arriba y respondiendo, independientemente
    de si el motor de NLP tiene o no datos cargados correctamente.
    """
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Configuración pensada para funcionar igual en local y en la mayoría
    # de plataformas de hosting gratuito: el puerto se toma de la variable
    # de entorno PORT si existe (Render la define automáticamente), y cae
    # a 5000 para desarrollo local si no está definida.
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
