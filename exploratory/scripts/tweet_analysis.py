"""Tweet Analysis and Graphs"""

from collections import Counter
from dataclasses import dataclass
from functools import cached_property

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
from pandas import DataFrame, notna

from exploratory.scripts.ptdf import ptdf
from exploratory.scripts.tweet import Tweet


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
            tweet = create_tweet(row)  # type: ignore
            obj_tweets.append(tweet)
        return ptdf(obj_tweets)

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
        figure(figsize=(20, 10))
        bar(elements, counts)
        xticks(rotation=80, ha="right")
        xlabel("Words")
        ylabel("Counts")
        title(f"Top 100 Elements in Counter in {self.dataset_name}")
        tight_layout()
        show()

    def plt_categorical_property(self, properties: list[str], pivot: str) -> None:
        """Plots the count of the specified property categorized by the pivot."""
        for property in properties:
            self.extra_data.groupby([property, pivot]).size().unstack().plot(kind="bar")
            xlabel(property)
            ylabel("Count")
            title(f"Count of {property} by {pivot}")
            xticks(rotation=45)
            show()
