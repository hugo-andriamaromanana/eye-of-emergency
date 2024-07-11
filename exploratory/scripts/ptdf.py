"""Converts a Sequence of Pydantic Objects to a Dataframe"""

from enum import Enum
from typing import Any, Sequence

from pandas import DataFrame
from pydantic import BaseModel


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


def ptdf(pydantic_objs: Sequence[BaseModel]) -> DataFrame:
    """Pydantic To DataFrame
    Converts a list of Pydantic objects to a DataFrame using attributes from
    get_attrs.
    """
    attrs = get_attrs(pydantic_objs[0]) - get_attrs(BaseModel)
    data = []
    for obj in pydantic_objs:
        row = create_row(obj, attrs)
        data.append(row)
    return DataFrame(data)
