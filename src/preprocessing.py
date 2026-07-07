"""
preprocessing.py
=================

Pipeline de preprocesamiento de texto en español para el agente
conversacional (RF-02).

Etapas aplicadas, en orden estricto:
    1. Normalización a minúsculas.
    2. Eliminación de tildes (á -> a) conservando la ñ.
    3. Eliminación de puntuación y caracteres especiales.
    4. Tokenización simple (split por espacios).
    5. Eliminación de stopwords en español.
    6. Stemming (SnowballStemmer de NLTK).

Decisión de diseño: stopwords embebidas en vez de nltk.corpus.stopwords
-------------------------------------------------------------------------
Se usa una lista de stopwords definida directamente en este archivo, en
lugar de `nltk.corpus.stopwords.words('spanish')`. La razón es puramente
operativa: ese corpus requiere `nltk.download('stopwords')` la primera vez
que se usa, lo cual implica escritura en disco y una conexión saliente a
los servidores de NLTK. En un plan gratuito de PythonAnywhere o en el
sistema de archivos efímero de Render, esa descarga puede fallar o no
persistir entre despliegues. Embeber la lista elimina esa dependencia sin
sacrificar cobertura para el dominio de este proyecto.

El stemmer (`SnowballStemmer`), en cambio, sí se importa de NLTK porque no
requiere descarga de datos externos: es un algoritmo puramente basado en
reglas (Porter/Snowball), compilado dentro del propio paquete.
"""

import re
import unicodedata
from typing import List

# ---------------------------------------------------------------------------
# Inicialización del stemmer con degradación segura
# ---------------------------------------------------------------------------
try:
    from nltk.stem.snowball import SnowballStemmer
    _stemmer = SnowballStemmer("spanish")
except Exception:
    # Si por algún motivo nltk no está instalado o falla la carga del
    # stemmer, se usa un "stemmer" identidad (no transforma la palabra).
    # Esto evita que todo el pipeline de PLN se caiga por un problema de
    # una sola dependencia opcional; el agente sigue funcionando, solo que
    # sin normalización morfológica.
    class _NullStemmer:
        def stem(self, word: str) -> str:
            return word

    _stemmer = _NullStemmer()

# ---------------------------------------------------------------------------
# Stopwords en español (subconjunto curado, suficiente para el dominio de
# admisión/nivelación; no pretende ser una lista lingüísticamente exhaustiva).
# ---------------------------------------------------------------------------
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
    """
    Elimina diacríticos (tildes) de un texto, conservando la ñ.

    Se usa la normalización Unicode NFKD, que descompone cada carácter
    acentuado en su letra base + un carácter de combinación de acento
    (ej. 'á' -> 'a' + '´'). Al filtrar los caracteres de combinación,
    queda solo la letra base.

    Args:
        texto: cadena de entrada, típicamente ya en minúsculas.

    Returns:
        El mismo texto sin tildes (ej. "admisión" -> "admision").
    """
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(caracter for caracter in nfkd if not unicodedata.combining(caracter))


def limpiar_texto(texto: str) -> str:
    """
    Normaliza el texto: minúsculas, sin tildes, sin puntuación ni caracteres
    especiales. Conserva letras, dígitos, la letra ñ y espacios simples.

    Esta función se usa tanto en el pipeline de preprocesamiento completo
    (`preprocesar`) como de forma independiente en `nlp_engine.py` para
    alimentar el vectorizador TF-IDF a nivel de caracteres, que necesita
    texto limpio pero SIN stemming (el stemming destruye subcadenas útiles
    para detectar errores tipográficos).

    Args:
        texto: cadena de entrada en cualquier combinación de mayúsculas/
            minúsculas, con o sin tildes/puntuación.

    Returns:
        Texto normalizado en minúsculas, sin tildes ni signos de puntuación,
        con espacios colapsados a uno solo.
    """
    texto = texto.lower()
    texto = quitar_tildes(texto)
    # \s conserva espacios; se descarta cualquier símbolo que no sea letra,
    # dígito, ñ o espacio (comas, signos de interrogación, emojis, etc.)
    texto = re.sub(r"[^a-z0-9ñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def tokenizar(texto: str) -> List[str]:
    """
    Tokenización simple por espacios en blanco.

    No se usa un tokenizador más sofisticado (ej. NLTK word_tokenize)
    porque, tras `limpiar_texto`, el texto ya no contiene puntuación que
    justifique reglas de tokenización más complejas; un split() estándar
    es suficiente y evita una dependencia adicional en tiempo de ejecución.

    Args:
        texto: texto ya normalizado por `limpiar_texto`.

    Returns:
        Lista de tokens (palabras). Lista vacía si el texto es vacío.
    """
    return texto.split() if texto else []


def eliminar_stopwords(tokens: List[str]) -> List[str]:
    """Filtra los tokens presentes en STOPWORDS_ES."""
    return [token for token in tokens if token not in STOPWORDS_ES]


def aplicar_stemming(tokens: List[str]) -> List[str]:
    """
    Reduce cada token a su raíz morfológica (stem) usando SnowballStemmer.

    Ejemplo de por qué esto ayuda a TF-IDF: "requisito", "requisitos" y
    "requeridos" comparten una raíz cercana, por lo que el stemming evita
    que TF-IDF los trate como dimensiones completamente independientes del
    vocabulario, mejorando la coincidencia semántica superficial entre la
    consulta del usuario y las utterances de entrenamiento.
    """
    return [_stemmer.stem(token) for token in tokens]


def preprocesar(texto: str) -> str:
    """
    Pipeline completo de preprocesamiento para representación por PALABRAS.

    Orden: limpieza -> tokenización -> eliminación de stopwords -> stemming.

    Esta es la función que alimenta el vectorizador TF-IDF de palabras en
    `nlp_engine.py`. Para el vectorizador de caracteres (parte del enfoque
    híbrido que tolera errores tipográficos), se usa en cambio
    `limpiar_texto` directamente, SIN este pipeline completo, ya que el
    stemming y la eliminación de stopwords no aportan valor a un análisis
    por n-gramas de caracteres y sí podrían eliminar información útil.

    Args:
        texto: consulta cruda del usuario.

    Returns:
        Cadena de tokens procesados, separados por espacios, lista para
        `TfidfVectorizer`. Cadena vacía si la entrada es vacía o solo
        contenía stopwords/ruido.
    """
    if not texto or not texto.strip():
        return ""
    texto = limpiar_texto(texto)
    tokens = tokenizar(texto)
    tokens = eliminar_stopwords(tokens)
    tokens = aplicar_stemming(tokens)
    return " ".join(tokens)


if __name__ == "__main__":
    # Demostración rápida del efecto del pipeline sobre ejemplos reales
    # del dominio (útil para ilustrar la pregunta "c" de la Justificación).
    ejemplos = [
        "¿Cuáles son los REQUISITOS de Admisión?",
        "Cómo creo mi cuenta de aspirante??",
        "cuando  son   las fechas de nivelación",
    ]
    for ejemplo in ejemplos:
        print(f"{ejemplo!r:55} -> {preprocesar(ejemplo)!r}")
