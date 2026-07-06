"""
Preprocesamiento de texto para el agente conversacional (RF-02).

Aplica, en este orden:
1. Normalización de minúsculas y eliminación de tildes/caracteres especiales.
2. Eliminación de signos de puntuación.
3. Tokenización simple.
4. Eliminación de stopwords en español.
5. Stemming básico (SnowballStemmer de NLTK, no requiere descarga de corpus).

Se usa una lista de stopwords embebida (en vez de nltk.stopwords) para que el
proyecto no dependa de `nltk.download(...)` al desplegarse en un servidor
(PythonAnywhere / Render), donde no siempre hay acceso de escritura a disco
o conexión saliente para descargar corpora.
"""

import re
import unicodedata

try:
    from nltk.stem.snowball import SnowballStemmer
    _stemmer = SnowballStemmer("spanish")
except Exception:
    # Si nltk no está disponible, se usa un "stemmer" identidad (no rompe el flujo).
    class _NullStemmer:
        def stem(self, word):
            return word
    _stemmer = _NullStemmer()

# Lista de stopwords en español (subconjunto amplio, suficiente para el dominio).
STOPWORDS_ES = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con",
    "contra", "cual", "cuando", "de", "del", "desde", "donde", "durante", "e",
    "el", "ella", "ellas", "ellos", "en", "entre", "era", "erais", "eran",
    "eras", "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta",
    "estaba", "estaban", "estamos", "estan", "estar", "este", "esto",
    "estos", "fue", "fueron", "fui", "fuimos", "ha", "hace", "haces",
    "hacia", "han", "has", "hasta", "hay", "la", "las", "le", "les", "lo",
    "los", "mas", "me", "mi", "mis", "mucho", "muchos", "muy", "nada",
    "ni", "no", "nos", "nosotros", "o", "os", "otra", "otras", "otro",
    "otros", "para", "pero", "poco", "por", "porque", "que", "quien",
    "quienes", "se", "sera", "seran", "si", "sin", "sobre", "sois", "somos",
    "son", "soy", "su", "sus", "tambien", "tan", "tanto", "te", "tenia",
    "tiene", "tienen", "todo", "todos", "tu", "tus", "un", "una", "uno",
    "unos", "y", "ya", "yo",
}


def quitar_tildes(texto: str) -> str:
    """Convierte caracteres acentuados a su forma base (á -> a, ñ se conserva)."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def limpiar_texto(texto: str) -> str:
    """Minúsculas, sin tildes, sin puntuación/caracteres especiales."""
    texto = texto.lower()
    texto = quitar_tildes(texto)
    # conserva letras, numeros, ñ y espacios
    texto = re.sub(r"[^a-z0-9ñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def tokenizar(texto: str) -> list:
    return texto.split() if texto else []


def eliminar_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS_ES]


def aplicar_stemming(tokens: list) -> list:
    return [_stemmer.stem(t) for t in tokens]


def preprocesar(texto: str) -> str:
    """
    Pipeline completo: limpieza -> tokenización -> stopwords -> stemming.
    Devuelve una cadena lista para ser vectorizada con TF-IDF.
    """
    if not texto or not texto.strip():
        return ""
    texto = limpiar_texto(texto)
    tokens = tokenizar(texto)
    tokens = eliminar_stopwords(tokens)
    tokens = aplicar_stemming(tokens)
    return " ".join(tokens)


if __name__ == "__main__":
    ejemplos = [
        "¿Cuáles son los REQUISITOS de Admisión?",
        "Cómo creo mi cuenta de aspirante??",
        "cuando  son   las fechas de nivelación",
    ]
    for e in ejemplos:
        print(f"{e!r:55} -> {preprocesar(e)!r}")
