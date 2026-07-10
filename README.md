# Asistente Virtual de Admisión y Nivelación — UG

Chatbot de intenciones basado en **PLN clásico** para responder preguntas
frecuentes sobre admisión y nivelación de la Universidad de Guayaquil.
Proyecto académico — Trabajo Parcial II, materia de Procesamiento de
Lenguaje Natural.

> El enunciado indica que **no se requiere despliegue en producción**.
> Los archivos de despliegue (Render/PythonAnywhere) incluidos aquí son un
> extra, no un requisito de la rúbrica.

## Arquitectura del motor de NLP

La detección de intención combina **dos representaciones TF-IDF en
paralelo** (ver `src/nlp_engine.py`), concatenadas antes de calcular
similitud coseno:

| Canal | Qué captura | Peso |
|---|---|---|
| TF-IDF por palabras (uni+bigramas), sobre texto con stopwords eliminadas y stemming | Significado léxico | 0.7 |
| TF-IDF por n-gramas de caracteres (3-5), sobre texto solo limpiado | Tolerancia a errores tipográficos ("nivelasion", "amision", "rejistro") | 0.3 |

Esto sigue siendo PLN clásico (ningún embedding ni modelo preentrenado):
es simplemente una segunda función de vectorización léxica, actuando como
red de seguridad ortográfica sobre la primera.

## Estructura del repositorio

```
ug-chatbot/
├── data/
│   ├── intents.json          # RF-01: 21 intenciones, utterances, respuestas y fuente/url
│   └── test_queries.csv      # RF-08: 50 consultas de prueba etiquetadas
├── src/
│   ├── preprocessing.py      # RF-02: limpieza, stopwords, stemming
│   ├── entities.py           # RF-05: fechas, cédula (con validación módulo 10), carreras, términos
│   ├── nlp_engine.py         # RF-03/04/06: TF-IDF híbrido, similitud coseno, fallback
│   ├── chatbot.py            # RF-07: interfaz de consola
│   ├── app.py                # RF-07: interfaz web (Flask)
│   └── evaluate.py           # RF-08: evaluación + barrido de umbrales
├── templates/
│   └── index.html            # Frontend del chat web
├── resultados_evaluacion.txt # Generado por evaluate.py (evidencia para el informe)
├── Informe_P2.pdf             # Informe escrito (3-4 páginas) requerido por el enunciado
├── wsgi.py                   # Punto de entrada para producción (Render)
├── Procfile                  # Comando de arranque para Render
├── render.yaml                # Configuración declarativa opcional de Render
├── .python-version            # Fija la versión de Python en Render
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
# genera ../resultados_evaluacion.txt con el reporte completo
```

## Resultados de evaluación

Sobre las 50 consultas de `data/test_queries.csv` (incluyendo variantes con
errores tipográficos, ambigüedad y consultas fuera de dominio), con el
umbral de producción `UMBRAL_CONFIANZA = 0.25`:

```
Accuracy : 0.980
F1-macro : 0.975
```

El barrido de umbrales (`evaluate.py`) confirma que todo el rango 0.20–0.40
da el mismo resultado; solo a partir de 0.45 desaparece un falso positivo
puntual fuera de dominio ("me puedes recomendar una laptop" → confianza
0.404 hacia `consultar_fechas_registro`). Se optó por 0.25 en vez de un
umbral más alto para minimizar el fallback en consultas válidas de
confianza moderada, aceptando ese único caso límite como parte de la
discusión de limitaciones (ver más abajo). Ver `resultados_evaluacion.txt`
para el detalle completo y el reporte por clase.

## Despliegue en producción

### Opción A: Render

1. Sube este repositorio a GitHub (público o accesible al docente).
2. En [render.com](https://render.com) → **New +** → **Web Service** → conecta el repo.
3. Render detecta `render.yaml` automáticamente, o configura manualmente:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn wsgi:app --bind 0.0.0.0:$PORT`
   - **Runtime:** Python 3 (fijado en `.python-version` como 3.11.9)
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
> arrancar el proceso (ver `src/app.py`), para que el ajuste de ambos
> vectorizadores TF-IDF no se repita en cada consulta.

## Justificación 

- **a) TF-IDF vs. bolsa de palabras:** el factor IDF reduce el peso de
  palabras muy frecuentes en todo el corpus (p. ej. "admision",
  "nivelacion") y resalta términos que distinguen una intención de otra.
- **b) Similitud coseno vs. distancia euclidiana:** el coseno normaliza por
  la longitud del vector, así una consulta corta y una utterance larga con
  las mismas palabras clave siguen siendo "cercanas".
- **c) Preprocesamiento antes de vectorizar:** ver `preprocessing.py` —
  "requisitos"/"requisito"/"REQUISITOS" se normalizan al mismo stem.
- **d) Umbral de confianza:** documentado y justificado con datos reales del
  barrido de `evaluate.py` (ver `nlp_engine.UMBRAL_CONFIANZA` y la sección
  "Resultados de evaluación" arriba).
- **e) Separar datos (JSON) de lógica (código):** permite ampliar o corregir
  intenciones —incluyendo su fuente oficial— sin tocar `nlp_engine.py`.
- **f) Manejo de errores:** `detectar_intencion()` en `nlp_engine.py`
  envuelve todo el flujo en `try/except` y cae a `fallback` ante texto
  vacío, no reconocido o cualquier excepción inesperada.

## Limitaciones observadas

- El canal de n-gramas de caracteres, que da tolerancia a errores
  tipográficos, también puede generar falsos positivos cuando una consulta
  fuera de dominio comparte subcadenas cortas con utterances de
  entrenamiento (ej. una entrada sin sentido que contiene el fragmento
  "signif" empareja parcialmente con "cupo aceptado ¿qué **signif**ica?").
- El set de prueba (50 consultas) es reducido frente al universo real de
  formas en que un estudiante puede escribir una pregunta; un umbral que
  luce óptimo aquí podría no generalizar igual ante un volumen mayor de
  tráfico real.
- La validación de cédula (`entities.validar_cedula_ecuador`) verifica
  estructura y dígito verificador, pero no consulta un padrón real: no
  puede confirmar que la cédula pertenezca efectivamente a un aspirante
  registrado.
- Intenciones semánticamente cercanas (p. ej. `agradecimiento` vs.
  `despedida`) pueden confundirse en frases mixtas, porque TF-IDF no
  captura intención pragmática, solo coincidencia léxica/ortográfica.
