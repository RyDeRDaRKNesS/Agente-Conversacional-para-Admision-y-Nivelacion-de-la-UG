"""
Motor del agente conversacional:
- Carga intents.json (RF-01)
- Vectoriza utterances con TF-IDF, uni+bigramas (RF-03)
- Detecta intención por similitud coseno (RF-04)
- Aplica umbral de confianza y fallback (RF-06)
- Extrae entidades (RF-05)

Diseñado para poder importarse tanto desde la interfaz de consola (chatbot.py)
como desde la interfaz web (app.py), sin duplicar lógica.
"""

import json
import os
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocesar
from entities import extraer_entidades

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTENTS_PATH = os.path.join(BASE_DIR, "data", "intents.json")

# Umbral de confianza: por debajo de este valor de similitud coseno,
# se considera que la consulta no fue reconocida (respuesta de fallback).
# Elegido empíricamente probando el set de evaluación (ver src/evaluate.py):
# valores más altos (0.4+) aumentaban los falsos "no reconocido" en frases
# cortas; valores más bajos (0.1) generaban falsos positivos entre intents
# parecidos (p. ej. nivelación vs. registro).
UMBRAL_CONFIANZA = 0.20


class MotorNLP:
    def __init__(self, intents_path: str = INTENTS_PATH, umbral: float = UMBRAL_CONFIANZA):
        self.umbral = umbral
        self.intents = self._cargar_intents(intents_path)
        self._construir_vectorizador()

    def _cargar_intents(self, path: str) -> list:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data["intents"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(f"No se pudo cargar el archivo de intents: {e}")

    def _construir_vectorizador(self):
        """Arma el corpus de utterances preprocesadas y ajusta el TF-IDF."""
        self.corpus = []          # utterances preprocesadas
        self.tag_por_utterance = []  # a qué intent pertenece cada fila del corpus

        for intent in self.intents:
            for utt in intent.get("utterances", []):
                texto_proc = preprocesar(utt)
                if texto_proc:
                    self.corpus.append(texto_proc)
                    self.tag_por_utterance.append(intent["tag"])

        if not self.corpus:
            raise RuntimeError("El corpus de utterances está vacío. Revisa data/intents.json")

        # TF-IDF con unigramas y bigramas, sobre matrices dispersas (RF-03)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.matriz_tfidf = self.vectorizer.fit_transform(self.corpus)

    def _respuesta_por_tag(self, tag: str) -> str:
        for intent in self.intents:
            if intent["tag"] == tag:
                return random.choice(intent["respuestas"])
        return self._respuesta_fallback()

    def _respuesta_fallback(self) -> str:
        for intent in self.intents:
            if intent["tag"] == "fallback":
                return random.choice(intent["respuestas"])
        return "Lo siento, no entendí. ¿Puedes reformular tu pregunta?"

    def detectar_intencion(self, consulta: str) -> dict:
        """
        Devuelve un diccionario con:
        - intent: tag detectado (o 'fallback')
        - confianza: score de similitud coseno máximo
        - texto_procesado: la consulta ya preprocesada (para depuración/informe)
        """
        try:
            if not consulta or not consulta.strip():
                return {"intent": "fallback", "confianza": 0.0, "texto_procesado": ""}

            texto_proc = preprocesar(consulta)
            if not texto_proc:
                return {"intent": "fallback", "confianza": 0.0, "texto_procesado": ""}

            vector_consulta = self.vectorizer.transform([texto_proc])
            similitudes = cosine_similarity(vector_consulta, self.matriz_tfidf)[0]

            idx_max = similitudes.argmax()
            score_max = float(similitudes[idx_max])

            if score_max < self.umbral:
                return {"intent": "fallback", "confianza": score_max, "texto_procesado": texto_proc}

            tag_detectado = self.tag_por_utterance[idx_max]
            return {"intent": tag_detectado, "confianza": score_max, "texto_procesado": texto_proc}

        except Exception as e:
            # Nunca debe caerse el agente por una entrada problemática (RF: manejo con try/except)
            return {"intent": "fallback", "confianza": 0.0, "texto_procesado": "", "error": str(e)}

    def responder(self, consulta: str) -> dict:
        """
        Punto de entrada principal: detecta intención, arma respuesta y extrae entidades.
        Devuelve un dict listo para usarse tanto en consola como en la API web.
        """
        resultado = self.detectar_intencion(consulta)
        respuesta = self._respuesta_por_tag(resultado["intent"])
        entidades = extraer_entidades(consulta) if consulta else {}

        return {
            "consulta": consulta,
            "intent": resultado["intent"],
            "confianza": round(resultado["confianza"], 4),
            "respuesta": respuesta,
            "entidades": entidades,
        }


if __name__ == "__main__":
    motor = MotorNLP()
    pruebas = [
        "Hola, buenas tardes",
        "cuales son los requisitos para inscribirme",
        "cuando empieza la nivelacion este periodo",
        "mi cedula es 0912345678, tengo cupo aceptado?",
        "asdkjaslkdj esto no significa nada",
    ]
    for p in pruebas:
        r = motor.responder(p)
        print(f"> {p}")
        print(f"  intent={r['intent']}  confianza={r['confianza']}")
        print(f"  respuesta: {r['respuesta']}")
        print(f"  entidades: {r['entidades']}")
        print()
