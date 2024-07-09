"""Text processing"""

from contractions import fix
from string import punctuation
from re import sub
from spacy import load as spacy_load


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


def lemmatize(txt: str) -> str:
    """Transform words to their root form"""
    eng_nlp = spacy_load("en_core_web_sm")
    doc = eng_nlp(txt)
    lemmatized_words = [token.lemma_ for token in doc]
    return " ".join(lemmatized_words)


def clean_txt(txt: str) -> str:
    """Cleans the text using the listed specs:
    - Punctuation removal
    - Number removal
    - Whitespaces removal
    - Expanded context (contractions)
    - Lemmatization (spacy)
    """
    wo_puncts = rm_punctuations(txt)
    wo_numbers = rm_numbers(wo_puncts)
    wo_whitespaces = rm_whitespaces(wo_numbers)
    expanded = expand_contractions(wo_whitespaces)
    lemmatized = lemmatize(expanded)
    return lemmatized
