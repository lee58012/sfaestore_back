import cv2
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import os

# 모델 로드
print("YOLO 모델 로드 중...")
model = YOLO('yolov8n.pt')

# 비디오 경로
VIDEO_PATH = "stored_videos/case2.mp4"
OUTPUT_PATH = "stored_videos/case2_detected.mp4"

# 계산대와 입구 구역 정의
ZONE_CHECKOUT = Polygon([(468, 120), (672, 110), (661, 469), (487, 472)])
ZONE_ENTRY = Polygon([(112, 131), (265, 136), (281, 439), (136, 471)])

# 비디오 열기
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"FPS: {fps}, Resolution: {width}x{height}")

# 출력 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

frame_count = 0
SKIP_FRAMES = 1  # 모든 프레임 분석

print("프레임 처리 중...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame_count += 1
    
    # 분석 건너뛰기
    if frame_count % SKIP_FRAMES == 0:
        # YOLO 추적
        results = model.track(frame, persist=True, classes=0, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 얼굴(머리) 좌표
                face_point = Point((x1 + x2) / 2, y1)
                
                # 바운딩 박스 색상 결정
                color = (0, 255, 0)  # 기본: 녹색
                
                # 계산대 구역 확인
                if ZONE_CHECKOUT.contains(face_point):
                    color = (0, 0, 255)  # 빨강색 (계산대)
                
                # 입구 구역 확인
                if ZONE_ENTRY.contains(face_point):
                    color = (255, 0, 0)  # 파랑색 (입구)
                
                # 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 계산대 구역 다각형 그리기 (파랑)
    checkout_points = [(527, 118), (666, 111), (646, 468), (496, 476)]
    checkout_pts = [tuple(p) for p in checkout_points]
    cv2.polylines(frame, [__import__('numpy').array(checkout_pts)], True, (255, 0, 0), 2)
    
    # 입구 구역 다각형 그리기 (녹색)
    entry_points = [(112, 131), (265, 136), (281, 439), (136, 471)]
    entry_pts = [tuple(p) for p in entry_points]
    cv2.polylines(frame, [__import__('numpy').array(entry_pts)], True, (0, 255, 0), 2)
    
    # 프레임 번호 표시
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 범례 표시
    cv2.putText(frame, "Red Box: Checkout Zone", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, "Green Box: Entry Zone", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "Blue Polygon: Checkout Area", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(frame, "Green Polygon: Entry Area", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 파일에 저장
    out.write(frame)
    
    if frame_count % 100 == 0:
        print(f"처리 완료: {frame_count} 프레임")

cap.release()
out.release()

print(f"✅ 완료! 저장 위치: {OUTPUT_PATH}")
print(f"총 프레임: {frame_count}")
