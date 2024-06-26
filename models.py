from pydantic import BaseModel
from typing import List

class SentenceCategory(BaseModel):
    sentence: str
    category: str

#class SentenceCategoryList(BaseModel):
#    data: List[SentenceCategory]

# class TrainingRequest(BaseModel):
#    data: List[SentenceCategory]
#    update_mode: str  # 'append' or 'replace'

class Item(BaseModel):
    text: str
