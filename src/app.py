"""
Interfaz web del agente conversacional (RF-07), y punto de entrada para
despliegue en producción (Render / PythonAnywhere).

Ejecución local:
    python app.py
Ejecución en producción (Render, con gunicorn):
    gunicorn src.app:app
"""

import os
import sys

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nlp_engine import MotorNLP  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# El motor se instancia una sola vez al arrancar el servidor (no en cada request).
motor = MotorNLP()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        mensaje = data.get("mensaje", "")
        resultado = motor.responder(mensaje)
        return jsonify(resultado)
    except Exception as e:
        # El agente nunca debe caerse ante una entrada problemática.
        return jsonify({
            "consulta": "",
            "intent": "fallback",
            "confianza": 0.0,
            "respuesta": "Ocurrió un problema procesando tu mensaje. ¿Puedes intentar de nuevo?",
            "entidades": {},
            "error": str(e),
        }), 200


@app.route("/health")
def health():
    """Endpoint simple para que Render/PythonAnywhere verifiquen que el servicio está vivo."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
