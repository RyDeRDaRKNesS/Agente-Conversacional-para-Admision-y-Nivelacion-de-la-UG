"""
nlp_engine.py
==============

Motor de PLN del agente conversacional. Responsable de:

    - Cargar intents.json                                   (RF-01)
    - Vectorizar utterances con TF-IDF                       (RF-03)
    - Detectar la intención por similitud coseno             (RF-04)
    - Aplicar umbral de confianza y respuesta de fallback     (RF-06)
    - Delegar la extracción de entidades a `entities.py`      (RF-05)

Estrategia de representación: TF-IDF híbrido (palabras + caracteres)
----------------------------------------------------------------------
La primera versión de este motor usaba un único `TfidfVectorizer` a nivel
de PALABRAS (uni + bigramas). Esa versión falla ante errores tipográficos
comunes en estudiantes escribiendo desde el celular, por ejemplo:

    "requicitos de amision"   (en vez de "requisitos de admisión")
    "komo me rejistro"        (en vez de "cómo me registro")
    "nivelasion"              (en vez de "nivelación")

Un TF-IDF por palabras trata cada palabra mal escrita como un token
completamente distinto y fuera de vocabulario, con similitud coseno nula
frente al vocabulario de entrenamiento. La solución -sin dejar de ser PLN
clásico y sin usar embeddings ni modelos preentrenados- es **combinar dos
representaciones TF-IDF**:

    1. TF-IDF por PALABRAS (uni+bigramas)   -> captura significado léxico.
    2. TF-IDF por N-GRAMAS DE CARACTERES     -> tolera errores de tipeo,
       porque palabras parecidas comparten muchos n-gramas de caracteres
       aunque difieran en una o dos letras (ej. "nivelasion" y
       "nivelacion" comparten la mayoría de sus trigramas).

Ambas matrices dispersas se concatenan horizontalmente (`scipy.sparse.hstack`)
antes de calcular la similitud coseno, de modo que el score final refleja
tanto la coincidencia de significado como la similitud ortográfica.
"""

import json
import os
import random
from typing import Dict, List

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from entities import extraer_entidades
from preprocessing import limpiar_texto, preprocesar

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTENTS_PATH = os.path.join(BASE_DIR, "data", "intents.json")

# ---------------------------------------------------------------------------
# Umbral de confianza (RF-06)
# ---------------------------------------------------------------------------
# Por debajo de este valor de similitud coseno, la consulta se considera
# "no reconocida" y se responde con el mensaje de fallback en lugar de
# forzar la intención más parecida (que podría ser una coincidencia débil
# y engañosa).
#
# Cómo se eligió: `src/evaluate.py` barre los valores [0.20, 0.25, 0.28,
# 0.30, 0.35, 0.40, 0.45, 0.50] contra las 47 consultas etiquetadas de
# data/test_queries.csv y reporta accuracy/F1-macro para cada uno (ver
# `resultados_evaluacion.txt`, generado automáticamente al ejecutar el
# script). Resultado observado:
#
#     Umbral   Accuracy   F1-macro
#     0.20-0.40   0.979     0.975   (1 falso positivo fuera de dominio)
#     0.45-0.50   1.000     1.000   (el falso positivo cae a fallback)
#
# El único error persistente en el rango 0.20-0.40 es una consulta fuera de
# dominio ("me puedes recomendar una laptop") que alcanza confianza 0.404
# por compartir el verbo "recomendar/registrar" a nivel de n-gramas de
# caracteres con utterances de `consultar_fechas_registro`. Se fijó el
# umbral en 0.42 -por encima de ese punto de quiebre, pero por debajo de
# 0.45- como margen de seguridad: es el valor más bajo que ya elimina ese
# falso positivo específico sin acercarse a la zona donde empezarían a
# perderse coincidencias válidas de confianza moderada (ver notas de
# limitaciones en el README sobre el tamaño del set de prueba).
UMBRAL_CONFIANZA = 0.42

# Peso relativo de cada representación al concatenar las matrices TF-IDF.
# El canal de palabras pesa más porque es el que mejor captura significado;
# el canal de caracteres actúa como una red de seguridad ortográfica.
PESO_PALABRAS = 0.7
PESO_CARACTERES = 0.3


class MotorNLP:
    """
    Encapsula todo el ciclo de vida del motor de detección de intenciones:
    carga de datos, ajuste de los vectorizadores TF-IDF y resolución de
    consultas en tiempo de inferencia.

    Se instancia UNA sola vez por proceso (ver `src/app.py` y
    `src/chatbot.py`), porque ajustar TF-IDF sobre el corpus de utterances
    tiene un costo que no debe repetirse en cada request.
    """

    def __init__(self, intents_path: str = INTENTS_PATH, umbral: float = UMBRAL_CONFIANZA):
        """
        Args:
            intents_path: ruta al archivo JSON con intenciones/utterances/
                respuestas (ver RF-01).
            umbral: umbral de confianza mínimo para aceptar una intención
                detectada; por debajo de este valor se responde fallback.
        """
        self.umbral = umbral
        self.intents = self._cargar_intents(intents_path)
        self._construir_vectorizadores()

    # ------------------------------------------------------------------
    # Carga de datos (RF-01)
    # ------------------------------------------------------------------
    def _cargar_intents(self, path: str) -> list:
        """
        Lee y valida el archivo JSON de intenciones.

        Se separa esta responsabilidad en un método propio (en vez de
        hacerlo inline en __init__) para que sea fácil de testear de forma
        aislada y para centralizar el manejo de errores de E/O y de
        formato del archivo de datos.
        """
        try:
            with open(path, "r", encoding="utf-8") as archivo:
                data = json.load(archivo)
            return data["intents"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as error:
            raise RuntimeError(f"No se pudo cargar el archivo de intents: {error}")

    # ------------------------------------------------------------------
    # Construcción de los vectorizadores TF-IDF (RF-03)
    # ------------------------------------------------------------------
    def _construir_vectorizadores(self) -> None:
        """
        Arma dos corpus paralelos a partir de las utterances de
        `intents.json` -uno preprocesado para el canal de palabras, otro
        solo limpiado (sin stopwords/stemming) para el canal de
        caracteres- y ajusta ambos `TfidfVectorizer`.

        Ambas listas (`corpus_palabras`, `corpus_caracteres`) y
        `tag_por_utterance` quedan alineadas índice a índice: la posición
        i-ésima de las tres corresponde a la misma utterance original.
        """
        self.corpus_palabras: List[str] = []
        self.corpus_caracteres: List[str] = []
        self.tag_por_utterance: List[str] = []

        for intent in self.intents:
            for utterance in intent.get("utterances", []):
                texto_palabras = preprocesar(utterance)
                texto_caracteres = limpiar_texto(utterance)

                # Solo se descarta la utterance si AMBAS representaciones
                # quedan vacías (ej. una utterance compuesta solo de
                # stopwords tras la limpieza).
                if texto_palabras or texto_caracteres:
                    self.corpus_palabras.append(texto_palabras)
                    self.corpus_caracteres.append(texto_caracteres)
                    self.tag_por_utterance.append(intent["tag"])

        if not self.corpus_palabras:
            raise RuntimeError("El corpus de utterances está vacío. Revisa data/intents.json")

        # Canal 1: TF-IDF por palabras, uni + bigramas (captura significado léxico).
        self.vectorizer_palabras = TfidfVectorizer(ngram_range=(1, 2))

        # Canal 2: TF-IDF por n-gramas de caracteres (3 a 5 caracteres,
        # con padding de límites de palabra vía analyzer="char_wb").
        # Este canal es el que da tolerancia a errores tipográficos: dos
        # palabras que difieren en una letra ("nivelacion" / "nivelasion")
        # comparten la mayoría de sus n-gramas de caracteres.
        self.vectorizer_caracteres = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))

        matriz_palabras = self.vectorizer_palabras.fit_transform(self.corpus_palabras)
        matriz_caracteres = self.vectorizer_caracteres.fit_transform(self.corpus_caracteres)

        # Cada matriz se pondera ANTES de concatenar, para que el canal de
        # palabras domine el score cuando ambos canales aportan señal, y el
        # canal de caracteres solo actúe como refuerzo ante errores de tipeo.
        matriz_palabras = matriz_palabras * PESO_PALABRAS
        matriz_caracteres = matriz_caracteres * PESO_CARACTERES

        # Concatenación horizontal de matrices dispersas: cada fila sigue
        # representando una utterance, pero ahora con columnas = columnas
        # del canal de palabras + columnas del canal de caracteres.
        self.matriz_tfidf = hstack([matriz_palabras, matriz_caracteres]).tocsr()

    # ------------------------------------------------------------------
    # Resolución de respuestas a partir de un tag de intención
    # ------------------------------------------------------------------
    def _respuesta_por_tag(self, tag: str) -> str:
        """Devuelve una respuesta aleatoria entre las definidas para `tag`."""
        for intent in self.intents:
            if intent["tag"] == tag:
                return random.choice(intent["respuestas"])
        return self._respuesta_fallback()

    def _respuesta_fallback(self) -> str:
        """Respuesta por defecto cuando no hay una intención 'fallback' definida."""
        for intent in self.intents:
            if intent["tag"] == "fallback":
                return random.choice(intent["respuestas"])
        return "Lo siento, no entendí. ¿Puedes reformular tu pregunta?"

    # ------------------------------------------------------------------
    # Detección de intención (RF-04) + fallback (RF-06)
    # ------------------------------------------------------------------
    def detectar_intencion(self, consulta: str) -> Dict:
        """
        Determina la intención más probable de una consulta en lenguaje
        natural, usando similitud coseno sobre la representación TF-IDF
        híbrida (palabras + caracteres).

        Todo el método está envuelto en un único bloque try/except: ante
        cualquier entrada problemática (texto vacío, encoding inesperado,
        excepción interna de scikit-learn) el método degrada de forma
        segura a "fallback" en lugar de propagar la excepción y tumbar el
        proceso del agente (requisito de robustez de la Sección 7-f).

        Args:
            consulta: texto crudo escrito por el usuario.

        Returns:
            Diccionario con:
                intent: tag detectado, o "fallback" si no se reconoce.
                confianza: score de similitud coseno máximo obtenido.
                texto_procesado: la consulta tras el pipeline de palabras
                    (útil para depuración y para el informe).
        """
        try:
            if not consulta or not consulta.strip():
                return {"intent": "fallback", "confianza": 0.0, "texto_procesado": ""}

            texto_palabras = preprocesar(consulta)
            texto_caracteres = limpiar_texto(consulta)

            if not texto_palabras and not texto_caracteres:
                return {"intent": "fallback", "confianza": 0.0, "texto_procesado": ""}

            vector_palabras = self.vectorizer_palabras.transform([texto_palabras]) * PESO_PALABRAS
            vector_caracteres = self.vectorizer_caracteres.transform([texto_caracteres]) * PESO_CARACTERES
            vector_consulta = hstack([vector_palabras, vector_caracteres]).tocsr()

            similitudes = cosine_similarity(vector_consulta, self.matriz_tfidf)[0]

            indice_max = similitudes.argmax()
            score_max = float(similitudes[indice_max])

            if score_max < self.umbral:
                return {
                    "intent": "fallback",
                    "confianza": score_max,
                    "texto_procesado": texto_palabras,
                }

            tag_detectado = self.tag_por_utterance[indice_max]
            return {
                "intent": tag_detectado,
                "confianza": score_max,
                "texto_procesado": texto_palabras,
            }

        except Exception as error:
            # Cualquier fallo inesperado se traduce en una respuesta segura
            # de fallback; el error se conserva en el dict para depuración
            # (no se muestra al usuario final).
            return {
                "intent": "fallback",
                "confianza": 0.0,
                "texto_procesado": "",
                "error": str(error),
            }

    # ------------------------------------------------------------------
    # Punto de entrada de alto nivel
    # ------------------------------------------------------------------
    def responder(self, consulta: str) -> Dict:
        """
        Orquesta el flujo completo para una consulta: detección de
        intención, selección de respuesta y extracción de entidades.

        Es el único método que deberían llamar los consumidores externos
        del motor (`chatbot.py` para consola, `app.py` para la API web),
        para evitar que la lógica de negocio quede duplicada en dos
        interfaces distintas.

        Args:
            consulta: texto crudo del usuario.

        Returns:
            Diccionario listo para serializar a JSON (API web) o imprimir
            en consola, con las claves: consulta, intent, confianza,
            respuesta, entidades.
        """
        resultado_deteccion = self.detectar_intencion(consulta)
        respuesta = self._respuesta_por_tag(resultado_deteccion["intent"])
        entidades = extraer_entidades(consulta) if consulta else {}

        return {
            "consulta": consulta,
            "intent": resultado_deteccion["intent"],
            "confianza": round(resultado_deteccion["confianza"], 4),
            "respuesta": respuesta,
            "entidades": entidades,
        }


if __name__ == "__main__":
    # Demostración manual rápida, incluyendo casos con errores tipográficos
    # para ilustrar el efecto del canal de caracteres.
    motor = MotorNLP()
    pruebas = [
        "Hola, buenas tardes",
        "requicitos de amision",          # error tipográfico intencional
        "komo me rejistro en admision",   # error tipográfico intencional
        "cuando empieza la nivelasion",   # error tipográfico intencional
        "mi cedula es 0912345678, tengo cupo aceptado?",
        "asdkjaslkdj esto no significa nada",
    ]
    for prueba in pruebas:
        resultado = motor.responder(prueba)
        print(f"> {prueba}")
        print(f"  intent={resultado['intent']}  confianza={resultado['confianza']}")
        print(f"  respuesta: {resultado['respuesta']}")
        print(f"  entidades: {resultado['entidades']}")
        print()
