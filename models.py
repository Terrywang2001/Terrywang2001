from pydantic import BaseModel
from typing import List

class SentenceCategory(BaseModel):
    sentence: str
    category: str

class SentenceCategoryList(BaseModel):
    data: List[SentenceCategory]

class Item(BaseModel):
    text: str
