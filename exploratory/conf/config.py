from dataclasses import dataclass
from enum import Enum, auto
from json import load as json_load
from pathlib import Path
from typing import Any

from loguru import logger


class NlpModel(str, Enum):
    """Used NLP Models"""

    SPACY = auto()
    NLTK = auto()


CONFIG_PATH = Path("exploratory/conf/config.json")


def load_json_from_path(config_path: str | Path) -> Any:
    with open(config_path) as config_file:
        return json_load(config_file)


@dataclass
class Config:
    SPACY_MODEL: str
    NLP_MODEL: NlpModel


def load_default_config() -> Config:
    """Loads default config from the config path"""
    conf = load_json_from_path(CONFIG_PATH)
    config = Config(conf["SPACY_MODEL"], NlpModel[conf["NLP_MODEL"]])
    return config

CONFIG: Config = load_default_config()


def load_config(config: Config | None) -> None:
    """Initializes the config"""
    if config is not None:
        global CONFIG
        logger.info("Custom config selected")
        CONFIG = config
