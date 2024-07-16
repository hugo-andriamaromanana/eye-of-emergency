"""Text cleaning with Spacy"""

from functools import cache

from nltk import download as nltk_dl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def rm_stop_words(txt: list[str]) -> list[str]:
    """Removes stop words from txt using ntlk"""
    stop_words = set(stopwords.words("english"))
    return [word for word in txt if word not in stop_words]


def lemmatize(txt: list[str]) -> list[str]:
    """Uses Nltk to remove stop words"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.strip()) for word in txt]


@cache
def nltk_clean(txt: str) -> list[str]:
    """Uses Ntlk to process the text through:
    - Tokenization
    - Stop words removal
    - Lemmatization
    """
    nltk_dl("stopwords")
    nltk_dl("punkt")
    nltk_dl("wordnet")

    tokenized = word_tokenize(txt)
    wo_stop_words = rm_stop_words(tokenized)
    lemmatized = lemmatize(wo_stop_words)
    return lemmatized
