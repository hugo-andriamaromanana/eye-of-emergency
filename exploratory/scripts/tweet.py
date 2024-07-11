"""Tweet cleaning"""

from functools import cached_property
from re import findall
from typing import Any

from pandera.typing import Series
from pydantic import BaseModel

_HASHTAG_PATTERN = r"#\S+"
_USERNAME_PATTERN = r"@\S+"
_EMAIL_PATTERN = r"\S*@\S*\s?"
_LINK_PATTERN = r'http.+?(?="|<|\s|$)'


class Tweet(BaseModel):
    """Namespace for handling tweets"""

    id: int
    keyword: str | None
    location: str | None
    txt: str
    target: bool

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
