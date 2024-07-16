"""Tweet cleaning"""

from functools import cached_property
from re import findall, sub
from string import punctuation
from typing import Any
from unicodedata import normalize

from conf.config import CONFIG
from contractions import fix  # type: ignore
from pandera.typing import Series
from pydantic import BaseModel
from scripts.nlp_cleaning import nlp_clean

_HASHTAG_PATTERN = r"#\S+"
_USERNAME_PATTERN = r"@\S+"
_EMAIL_PATTERN = r"\S*@\S*\s?"
_LINK_PATTERN = r'http.+?(?="|<|\s|$)'
_UNICODE_PATTERN = r"[^\x00-\x7F]+"


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


def rm_unicodes(txt: str) -> str:
    """Removes unicode characters and escape sequences"""
    cleaned_text = (
        normalize("NFKD", txt).encode("ascii", "ignore").decode("utf-8", "ignore")
    )
    return sub(_UNICODE_PATTERN, "", cleaned_text)


def expand_contractions(txt: str) -> str:
    """Expands contrations from a text"""
    return str(fix(txt))


class Tweet(BaseModel):
    """Namespace for handling tweets"""

    id: int
    keyword: str | None
    location: str | None
    txt: str
    target: bool

    @property
    def tokenized_text(self) -> list[str]:
        """Cleans the text using the listed specs:
        - Lowercases and strips the entire string
        - Punctuation removal
        - Number removal
        - Whitespaces removal
        - Expanded context (contractions)
        - Text Processing (NLP)
        """
        lower = self.txt.lower().strip()
        wo_puncts = rm_punctuations(lower)
        wo_numbers = rm_numbers(wo_puncts)
        wo_whitespaces = rm_whitespaces(wo_numbers)
        wo_unicodes = rm_unicodes(wo_whitespaces)
        expanded = expand_contractions(wo_unicodes)
        nlp_cleaned = nlp_clean(CONFIG.NLP_MODEL, expanded)
        return nlp_cleaned

    @property
    def cleaned_txt(self) -> str:
        """Concatenates txt with spacespaces as string"""
        return " ".join(self.tokenized_text)

    @property
    def txt_len(self) -> int:
        """Nb of total characters in txt"""
        return len(self.txt)

    @property
    def nb_of_words(self) -> int:
        """Nb of total whitespace seperated strings"""
        return len([self.txt.split(" ")])

    @cached_property
    def hashtags(self) -> list[str]:
        """All hashtags contained in the txt"""
        return findall(_HASHTAG_PATTERN, self.txt)

    @cached_property
    def urls(self) -> list[str]:
        """ALl urls contained in the txt"""
        return findall(_LINK_PATTERN, self.txt)

    @cached_property
    def usernames(self) -> list[str]:
        """All usernames contained in the txt"""
        return findall(_USERNAME_PATTERN, self.txt)

    @cached_property
    def emails(self) -> list[str]:
        """All emails countained in the txt"""
        return findall(_EMAIL_PATTERN, self.txt)

    @property
    def has_url(self) -> bool:
        """If txt contains links"""
        return self.urls != []

    @property
    def has_hashtag(self) -> bool:
        """Returns whether the txt has hashtags"""
        return self.hashtags != []

    @property
    def has_username(self) -> bool:
        """Returns whether the txt has a username mentionned"""
        return self.usernames != []

    @property
    def has_email(self) -> bool:
        """Returns wheter the txt has a mail"""
        return self.emails != []


def create_tweet(row: Series[Any]) -> Tweet:
    """Abstraction of tweet creation"""
    target = True if row["target"] == 1 else False
    return Tweet(
        id=row["id"],
        keyword=row["keyword"],
        location=row["location"],
        txt=row["text"],
        target=target,
    )
