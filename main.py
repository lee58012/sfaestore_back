from fastapi import FastAPI, UploadFile, File
import shutil
import os
from logic import process_video

app = FastAPI()
os.makedirs("temp_videos", exist_ok=True)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    # 1. 파일 저장
    file_location = f"temp_videos/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"분석 시작: {file.filename}")

    # 2. 분석 수행 (오래 걸릴 수 있음)
    analysis_results = process_video(file_location)
    
    print(f"분석 완료: {len(analysis_results)}건 발견")

    # 3. 결과 반환
    return {"status": "success", "data": analysis_results}