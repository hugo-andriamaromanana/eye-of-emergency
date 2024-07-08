"""Tweet cleaning"""

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from re import findall
from typing import Any, Sequence

from matplotlib.pyplot import (
    bar,
    figure,
    show,
    tight_layout,
    title,
    xlabel,
    xticks,
    ylabel,
)
from pandas import DataFrame, Series, notna
from pydantic import BaseModel, Field


def get_attrs(obj: object) -> set[str]:
    """Gets all the attributes, and properties of an object
    excluding default object attributes
    """
    attrs = set()
    for attr in dir(obj):
        if not (attr.startswith("__") or attr.startswith("_")):
            attrs.add(attr)
    return attrs


def create_row(obj: object, attrs: set[str]) -> dict[str, Any]:
    """Creates row, ommit enum types"""
    row = {}
    for attr in attrs:
        att = getattr(obj, attr)
        if isinstance(att, Enum):
            att = att.name
        row[attr] = att
    return row


def pydantic_to_df(pydantic_objs: Sequence[BaseModel]) -> DataFrame:
    """Converts a list of Pydantic objects to a DataFrame using attributes from
    get_attrs.
    """
    attrs = get_attrs(pydantic_objs[0]) - get_attrs(BaseModel)
    data = []
    for obj in pydantic_objs:
        row = create_row(obj, attrs)
        data.append(row)
    return DataFrame(data)


_HASHTAG_PATTERN = r"#\S+"
_USERNAME_PATTERN = r"@\S+"
_EMAIL_PATTERN = r"\S*@\S*\s?"
_LINK_PATTERN = r'http.+?(?="|<|\s|$)'


class Tweet(BaseModel):
    """Namespace for handling tweets"""

    id: int = Field(compare=False)
    keyword: str | None = Field(compare=False)
    location: str | None = Field(compare=False)
    txt: str
    target: bool = Field(compare=False)

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


@dataclass
class TweetsAnalysis:
    """Quick useful TweetAnalysis namespace for plotting"""

    dataset_name: str
    raw_data: DataFrame

    @cached_property
    def extra_data(self) -> DataFrame:
        """Returns a df with extra labels"""
        df_none = self.raw_data.where(notna(self.raw_data), None)
        obj_tweets: list[Tweet] = []
        for _, row in df_none.iterrows():
            tweet = create_tweet(row)
            obj_tweets.append(tweet)
        return pydantic_to_df(obj_tweets)

    def plt_word_occs(self) -> None:
        """Plots the nb of highest occs in a dataframe"""
        all_text = " ".join([i for i in self.raw_data["text"]])
        words = all_text.split(" ")
        occs = Counter(words)
        del occs[" "]
        del occs["-"]
        del occs[""]
        top_100_elements = occs.most_common(100)
        elements, counts = zip(*top_100_elements)
        figure(figsize=(15, 7))
        bar(elements, counts)
        xticks(rotation=80, ha="right")
        xlabel("Words")
        ylabel("Counts")
        title(f"Top 100 Elements in Counter in {self.dataset_name}")
        tight_layout()
        show()

    def plot_categorical_property(self, properties: list[str], pivot: str) -> None:
        """Plots the count of the specified property categorized by the pivot."""
        for property in properties:
            grouped = self.extra_data.groupby([property, pivot]).size().unstack()
            ax = grouped.plot(kind="bar", stacked=False)
            ax.set_xlabel(property)
            ax.set_ylabel("Count")
            ax.set_title(f"Count of {property} by {pivot}")
            xticks(rotation=45)
            show()
