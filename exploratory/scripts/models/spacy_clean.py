"""Text cleaning with Spacy"""

from functools import cache
from sys import exit as sys_exit

from loguru import logger
from spacy import load as spacy_load
from spacy.language import Language
from spacy.tokens import Doc

@cache
def load_spacy_model(model: str) -> Language:
    """Loads the designed spacy model
    Ensures a single write on SPACY_ENG_LOAD
    """
    try:
        load = spacy_load(model)
        return load
    except Exception as e:
        logger.critical(f"Couldn't load model due to:\n{e}")
        sys_exit(0)
    


def get_doc_from_loader(txt: str, loader: Language) -> Doc:
    """Unpacks None, uses the model to get the doc obj"""
    return loader(txt)


@cache
def spacy_lemmatize(doc: Doc) -> list[str]:
    """Transform words to their root form"""
    return [token.lemma_ for token in doc]


@cache
def spacy_rm_stop_words(doc: Doc) -> list[str]:
    """Removes stop word from text"""
    return [token.text for token in doc if not token.is_stop]


def spacy_clean(txt: str) -> list[str]:
    """Uses Spacy to process the text through:
    -Lemmatization
    -Stop word removal
    """
    loader = load_spacy_model("en_core_web_sm")
    doc = get_doc_from_loader(txt, loader)
    lemmatized = spacy_lemmatize(doc)
    back_to_text = " ".join(lemmatized)
    lemma_doc = get_doc_from_loader(back_to_text, loader)
    wo_stop_words = spacy_rm_stop_words(lemma_doc)
    return wo_stop_words
