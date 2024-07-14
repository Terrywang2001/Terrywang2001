from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
import jieba
import os

app = FastAPI()

# 加载自定义词典
custom_dict_path = "custom_dict.txt"
if os.path.exists(custom_dict_path):
    jieba.load_userdict(custom_dict_path)

@app.post("/segment/")
async def segment_text(text: str):
    """
    对输入的文本进行分词
    """
    try:
        segmented_text = jieba.lcut(text)
        return {"original_text": text, "segmented_text": segmented_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_to_dict/")
async def add_to_dict(word: str, freq: Optional[int] = None, tag: Optional[str] = None):
    """
    添加一个词到自定义词典
    """
    try:
        if freq and tag:
            jieba.add_word(word, freq=freq, tag=tag)
        elif freq:
            jieba.add_word(word, freq=freq)
        else:
            jieba.add_word(word)
        
        with open(custom_dict_path, "a", encoding="utf-8") as f:
            if freq and tag:
                f.write(f"{word} {freq} {tag}\n")
            elif freq:
                f.write(f"{word} {freq}\n")
            else:
                f.write(f"{word}\n")

        return {"message": f"Word '{word}' added to custom dictionary successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove_from_dict/")
async def remove_from_dict(word: str):
    """
    从自定义词典中移除一个词
    """
    try:
        jieba.del_word(word)
        
        if os.path.exists(custom_dict_path):
            with open(custom_dict_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            with open(custom_dict_path, "w", encoding="utf-8") as f:
                for line in lines:
                    if not line.startswith(word + " "):
                        f.write(line)
        
        return {"message": f"Word '{word}' removed from custom dictionary successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
