from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import jieba
import jieba.analyse
import os

app = FastAPI()

# 加载自定义词典
custom_dict_path = "custom_dict.txt"
if os.path.exists(custom_dict_path):
    jieba.load_userdict(custom_dict_path)

class TextRequest(BaseModel):
    text: str

class AddWordRequest(BaseModel):
    word: str
    freq: Optional[int] = None
    tag: Optional[str] = None

class RemoveWordRequest(BaseModel):
    word: str

class KeywordRequest(BaseModel):
    text: str
    topK: Optional[int] = 5

@app.post("/segment/")
async def segment_text(request: TextRequest):
    """
    对输入的文本进行分词
    """
    try:
        segmented_text = jieba.lcut(request.text)
        return {"original_text": request.text, "segmented_text": segmented_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_to_dict/")
async def add_to_dict(request: AddWordRequest):
    """
    添加一个词到自定义词典
    """
    try:
        if request.freq and request.tag:
            jieba.add_word(request.word, freq=request.freq, tag=request.tag)
        elif request.freq:
            jieba.add_word(request.word, freq=request.freq)
        else:
            jieba.add_word(request.word)
        
        with open(custom_dict_path, "a", encoding="utf-8") as f:
            if request.freq and request.tag:
                f.write(f"{request.word} {request.freq} {request.tag}\n")
            elif request.freq:
                f.write(f"{request.word} {request.freq}\n")
            else:
                f.write(f"{request.word}\n")

        return {"message": f"Word '{request.word}' added to custom dictionary successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove_from_dict/")
async def remove_from_dict(request: RemoveWordRequest):
    """
    从自定义词典中移除一个词
    """
    try:
        jieba.del_word(request.word)
        
        if os.path.exists(custom_dict_path):
            with open(custom_dict_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            with open(custom_dict_path, "w", encoding="utf-8") as f:
                for line in lines:
                    if not line.startswith(request.word + " "):
                        f.write(line)
        
        return {"message": f"Word '{request.word}' removed from custom dictionary successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_keywords/")
async def extract_keywords(request: KeywordRequest):
    """
    提取关键词
    """
    try:
        keywords = jieba.analyse.extract_tags(request.text, topK=request.topK)
        return {"original_text": request.text, "keywords": keywords}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
