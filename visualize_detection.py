import cv2
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import os
import numpy as np

# 모델 로드
print("YOLO 모델 로드 중...")
model = YOLO('yolov8n.pt')

# 비디오 경로
VIDEO_PATH = "stored_videos/case3.mp4"
OUTPUT_PATH = "stored_videos/case3_detected.mp4"

# 계산대와 입구 구역 정의
ZONE_CHECKOUT = Polygon([(727, 185), (985, 160), (978, 713), (733, 716)])
ZONE_ENTRY = Polygon([(167, 204), (394, 210), (424, 650), (213, 705)])

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
        # YOLO 추적 (결과 안전 검사)
        results = model.track(frame, persist=True, classes=0, verbose=False)
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu()
            # 트랙 ID가 있을 수도 있고 없을 수도 있음 -> 안전하게 처리
            ids = None
            if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                try:
                    ids = results[0].boxes.id.int().cpu().tolist()
                except Exception:
                    ids = None

            for i, box in enumerate(boxes):
                track_id = ids[i] if ids is not None and i < len(ids) else -1
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
    checkout_points = [(727, 185), (985, 160), (978, 713), (733, 716)]
    checkout_pts = [tuple(p) for p in checkout_points]
    cv2.polylines(frame, [np.array(checkout_pts)], True, (255, 0, 0), 2)
    
    # 입구 구역 다각형 그리기 (녹색)
    entry_points = [(167, 204), (394, 210), (424, 650), (213, 705)]
    entry_pts = [tuple(p) for p in entry_points]
    cv2.polylines(frame, [np.array(entry_pts)], True, (0, 255, 0), 2)
    
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
