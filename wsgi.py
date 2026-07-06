"""
Punto de entrada WSGI para servidores de producción (Render con gunicorn).

Render ejecuta: gunicorn wsgi:app
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from app import app  # noqa: E402,F401

if __name__ == "__main__":
    app.run()
