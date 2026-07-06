# Asistente Virtual de Admisión y Nivelación — UG

Chatbot de intenciones basado en **PLN clásico** (TF-IDF + similitud coseno) para
responder preguntas frecuentes sobre admisión y nivelación de la Universidad de
Guayaquil. Proyecto académico — Trabajo Parcial II, materia de Procesamiento de
Lenguaje Natural.

> ⚠️ Nota: el enunciado del trabajo indica que **no se requiere despliegue en
> producción**. Los archivos de despliegue (Render/PythonAnywhere) incluidos
> aquí son un extra, no un requisito de la rúbrica.

## Estructura del repositorio

```
ug-chatbot/
├── data/
│   ├── intents.json        # RF-01: intenciones, utterances y respuestas
│   └── test_queries.csv     # RF-08: set de prueba (25 consultas etiquetadas)
├── src/
│   ├── preprocessing.py     # RF-02: limpieza, stopwords, stemming
│   ├── entities.py          # RF-05: extracción de entidades (regex)
│   ├── nlp_engine.py        # RF-03/04/06: TF-IDF, similitud coseno, fallback
│   ├── chatbot.py           # RF-07: interfaz de consola
│   ├── app.py               # RF-07: interfaz web (Flask)
│   └── evaluate.py          # RF-08: evaluación (accuracy, F1-macro)
├── templates/
│   └── index.html           # Frontend del chat web
├── wsgi.py                  # Punto de entrada para producción (Render)
├── Procfile                 # Comando de arranque para Render
├── render.yaml               # Configuración declarativa opcional de Render
└── requirements.txt
```

## Ejecución local

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Opción 1 — Consola:**
```bash
cd src
python chatbot.py
```

**Opción 2 — Interfaz web local:**
```bash
cd src
python app.py
# abrir http://127.0.0.1:5000
```

**Evaluación (RF-08):**
```bash
cd src
python evaluate.py
```

## Despliegue en producción

### Opción A: Render

1. Sube este repositorio a GitHub (público o accesible al docente).
2. En [render.com](https://render.com) → **New +** → **Web Service** → conecta el repo.
3. Render detecta `render.yaml` automáticamente, o configura manualmente:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn wsgi:app --bind 0.0.0.0:$PORT`
   - **Runtime:** Python 3
4. Despliega. La URL pública (`https://tu-app.onrender.com`) sirve la interfaz web.

### Opción B: PythonAnywhere

1. Sube el repo o clónalo desde una consola Bash de PythonAnywhere:
   ```bash
   git clone <url-de-tu-repo>
   cd ug-chatbot
   pip install --user -r requirements.txt
   ```
2. Ve a la pestaña **Web** → **Add a new web app** → **Manual configuration** → Python 3.10+.
3. Edita el archivo WSGI que PythonAnywhere genera automáticamente
   (`/var/www/tu_usuario_pythonanywhere_com_wsgi.py`) y reemplaza su contenido por:
   ```python
   import sys
   path = '/home/tu_usuario/ug-chatbot'
   if path not in sys.path:
       sys.path.insert(0, path)
   path_src = '/home/tu_usuario/ug-chatbot/src'
   if path_src not in sys.path:
       sys.path.insert(0, path_src)

   from app import app as application
   ```
4. En la pestaña **Web**, define la ruta de **Source code** y **Working directory**
   como `/home/tu_usuario/ug-chatbot`.
5. Click en **Reload**. La app queda disponible en `https://tu_usuario.pythonanywhere.com`.

> En ambos casos, el motor NLP (`MotorNLP`) se instancia **una sola vez** al
> arrancar el proceso (ver `src/app.py`), no en cada request, para que el
> cálculo de TF-IDF no se repita en cada consulta.

## Justificación (Sección 7 del enunciado) — guía rápida

- **a) TF-IDF vs. bolsa de palabras:** ver `nlp_engine.py` — el factor IDF
  reduce el peso de palabras muy frecuentes en todo el corpus (p. ej. "admision",
  "nivelacion") y resalta términos que distinguen una intención de otra.
- **b) Similitud coseno vs. distancia euclidiana:** el coseno normaliza por la
  longitud del vector, así una consulta corta ("cuando nivelacion") y una
  utterance larga con las mismas palabras clave siguen siendo "cercanas".
- **c) Preprocesamiento antes de vectorizar:** ver `preprocessing.py` — por
  ejemplo, "requisitos" / "requisito" / "REQUISITOS" se normalizan al mismo
  stem, evitando que TF-IDF los trate como tokens distintos.
- **d) Umbral de confianza:** documentado como constante `UMBRAL_CONFIANZA` en
  `nlp_engine.py`, elegido probando el set de evaluación (`evaluate.py`). Sin
  umbral, cualquier consulta fuera de dominio recibiría la respuesta de la
  intención "más parecida" aunque la similitud fuera casi nula.
- **e) Separar datos (JSON) de lógica (código):** permite ampliar o corregir
  intenciones sin tocar `nlp_engine.py`, y facilita que otra persona del grupo
  edite el contenido sin escribir Python.
- **f) Manejo de errores:** `detectar_intencion()` en `nlp_engine.py` envuelve
  todo el flujo en `try/except` y cae a `fallback` ante texto vacío, no
  reconocido o cualquier excepción inesperada, sin detener el proceso.

## Limitaciones observadas (ver salida de `evaluate.py`)

- Consultas muy cortas con una sola palabra de alto IDF pueden generar falsos
  positivos (p. ej. "significa" comparte stem con una utterance de otra
  intención, elevando su similitud aunque el resto de la frase no coincida).
- Intenciones semánticamente cercanas (p. ej. `agradecimiento` vs.
  `despedida`) a veces se confunden porque comparten vocabulario superficial;
  un enfoque puramente léxico como TF-IDF no captura la diferencia de
  intención pragmática entre "gracias" y "listo, gracias, adiós".
- El umbral fijo (0.20) es un compromiso: subirlo reduce falsos positivos pero
  aumenta las consultas válidas que caen a fallback.
