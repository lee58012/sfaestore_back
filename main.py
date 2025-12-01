from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import os
import json
from datetime import datetime
from logic import process_video

app = FastAPI()

# 폴더 생성
os.makedirs("temp_videos", exist_ok=True)
os.makedirs("stored_videos", exist_ok=True) # 이상행동 영상 저장소

# [중요] stored_videos 폴더를 외부에서 접속 가능하게 설정 (영상 재생용)
app.mount("/videos", StaticFiles(directory="stored_videos"), name="videos")

# 간이 DB 파일 경로
DB_FILE = "db.json"

# DB 읽기 함수
def load_db():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# DB 쓰기 함수
def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

@app.get("/")
def read_root():
    return {"message": "Safestore AI Server Running!"}

# 1. 앱에서 리스트 요청 시 저장된 기록 반환
@app.get("/logs")
def get_logs():
    return load_db()

# 2. 영상 업로드 및 분석
@app.post("/upload")
def upload_video(file: UploadFile = File(...)):
    # 파일 임시 저장
    temp_path = f"temp_videos/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"분석 시작: {file.filename}")
    
    # AI 분석 수행
    analysis_results = process_video(temp_path)
    
    saved_info = None
    
    # [핵심 로직] 이상행동이 발견되면 영상과 로그를 저장
    if len(analysis_results) > 0:
        # 1. 영상 파일을 보관소로 이동 (파일명에 시간 추가하여 중복 방지)
        timestamp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"{timestamp_name}_{file.filename}"
        saved_path = f"stored_videos/{saved_filename}"
        
        shutil.move(temp_path, saved_path) # 파일 이동
        
        # 2. DB에 기록 추가
        # 영상 URL 생성 (ngrok 주소는 앱에서 앞에 붙임)
        video_url_path = f"/videos/{saved_filename}"
        
        # 첫 번째 감지된 시간과 내용을 대표로 저장 (필요시 리스트 전체 저장 가능)
        representative_log = analysis_results[0]
        
        new_record = {
            "timestamp": representative_log["timestamp"], # 영상 내 발생 시간 (예: 00:15)
            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # 실제 발생 날짜
            "type": representative_log["type"],
            "videoUrl": video_url_path # 영상 재생 주소
        }
        
        # DB 저장
        current_db = load_db()
        current_db.append(new_record)
        save_db(current_db)
        
        saved_info = "이상행동 감지됨! 영상이 저장되었습니다."
        print(f"저장 완료: {saved_path}")
        
    else:
        # 이상행동 없으면 임시 파일 삭제
        os.remove(temp_path)
        saved_info = "이상행동 없음. 영상이 삭제되었습니다."
        print("이상행동 없음. 삭제 완료.")

    return {
        "status": "success", 
        "message": saved_info,
        "data": analysis_results, # 분석 결과 상세
        "saved_video_url": f"/videos/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}" if len(analysis_results) > 0 else None
    }
# 3. 특정 로그 및 영상 삭제
@app.delete("/logs/{filename}")
def delete_log(filename: str):
    current_db = load_db()
    
    # 1. DB에서 해당 파일명을 가진 기록 찾기
    # videoUrl 예시: "/videos/20240530_video.mp4" -> "20240530_video.mp4"만 비교
    updated_db = [log for log in current_db if filename not in log["videoUrl"]]
    
    # 만약 개수가 줄어들었다면 (삭제된 게 있다면)
    if len(current_db) > len(updated_db):
        save_db(updated_db) # DB 업데이트
        
        # 2. 실제 파일 삭제
        file_path = f"stored_videos/{filename}"
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"파일 삭제됨: {file_path}")
        
        return {"status": "success", "message": "삭제 완료"}
    
    return {"status": "error", "message": "파일을 찾을 수 없음"}