from sys import exit as sys_exit
from typing import Callable, TypeAlias

from loguru import logger

from conf.config import NlpModel
from scripts.models.nltk_clean import nltk_clean
from scripts.models.spacy_clean import spacy_clean

CleaningFunc: TypeAlias = Callable[[str], list[str]]

CLEANING_MAP: dict[NlpModel, CleaningFunc] = {
    NlpModel.NLTK: nltk_clean,
    NlpModel.SPACY: spacy_clean,
}


def nlp_clean(model: NlpModel, txt: str) -> list[str]:
    """Cleans the text using an nlp model. See CLEANING_MAP"""
    try:
        cleaned = CLEANING_MAP[model](txt)
        return cleaned
    except KeyError as key_err:
        logger.critical(f"NLP Model cleaning not implimented:\n{key_err}")
        sys_exit(0)
    except Exception as e:
        logger.warning(f"Failed to clean text with {model} due to:\n{e}")
        sys_exit(0)
