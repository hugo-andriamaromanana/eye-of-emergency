"""Text processing"""

from re import sub
from string import punctuation

from contractions import fix  # type: ignore

from exploratory.conf.config import CONFIG
from exploratory.scripts.nlp_cleaning import nlp_clean


def rm_punctuations(txt: str) -> str:
    """Removes punctuations from a text"""
    trans_table = str.maketrans("", "", punctuation)
    text = txt.translate(trans_table)
    return text


def rm_numbers(txt: str) -> str:
    """Removes numbers from a text"""
    re = r"\d+"
    return sub(re, "", txt)


def rm_whitespaces(txt: str) -> str:
    """Removes extra whitespaces, maybe due to typos"""
    return " ".join(txt.split())


def expand_contractions(txt: str) -> str:
    """Expands contrations from a text"""
    return str(fix(txt))


def clean_txt(txt: str) -> list[str]:
    """Cleans the text using the listed specs:
    - Lowercases and strips the entire string
    - Punctuation removal
    - Number removal
    - Whitespaces removal
    - Expanded context (contractions)
    - Lemmatization (spacy)
    """
    lower = txt.lower().strip()
    wo_puncts = rm_punctuations(lower)
    wo_numbers = rm_numbers(wo_puncts)
    wo_whitespaces = rm_whitespaces(wo_numbers)
    expanded = expand_contractions(wo_whitespaces)
    npl_cleaned = nlp_clean(CONFIG.NLP_MODEL, expanded)
    return npl_cleaned