r"""
``cotk.wordvector`` provides classes and functions downloading and
loading wordvector automatically.
"""

from .wordvector import WordVector
from .gloves import Glove
from .fasttext import FastText

__all__ = ["WordVector", "Glove", "FastText"]
