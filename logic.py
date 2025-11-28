import cv2
import time
from ultralytics import YOLO
from shapely.geometry import Point, Polygon

# 모델 로드
model = YOLO('yolov8n.pt')

# 구역 설정 (영상 해상도에 맞춰 좌표 수정 필수)
# 예: 1920x1080 해상도 기준 임의 좌표
ZONE_CHECKOUT = Polygon([(800, 400), (1200, 400), (1200, 800), (800, 800)])
ZONE_ENTRY = Polygon([(0, 800), (400, 800), (400, 1080), (0, 1080)])

def process_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 레이트 확인
    if fps == 0: fps = 30 # 기본값

    customers = {}
    logs = [] # 앱으로 보낼 결과 리스트

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # 현재 영상의 타임스탬프 (초 단위)
        current_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # YOLO 트래킹
        results = model.track(frame, persist=True, classes=0, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                foot_point = Point((x1 + x2) / 2, y2)

                # 1. 입장 (Entry)
                if track_id not in customers:
                    if ZONE_ENTRY.contains(foot_point):
                        customers[track_id] = {
                            "entry_time": current_video_time,
                            "checkout_start": None,
                            "checkout_duration": 0,
                            "status": "SHOPPING"
                        }

                # 2. 계산대 체류 (Checkout)
                if track_id in customers:
                    if ZONE_CHECKOUT.contains(foot_point):
                        if customers[track_id]["checkout_start"] is None:
                            customers[track_id]["checkout_start"] = current_video_time
                        
                        # 프레임 단위 시간 누적
                        customers[track_id]["checkout_duration"] += (1.0 / fps)
                    else:
                        customers[track_id]["checkout_start"] = None

                    # 3. 퇴장 및 판단 (Exit)
                    # 입장 구역에 다시 왔고, 들어온지 5초 이상 지났을 때
                    if ZONE_ENTRY.contains(foot_point) and (current_video_time - customers[track_id]["entry_time"] > 5):
                        if customers[track_id]["status"] != "EXITED":
                            duration = customers[track_id]["checkout_duration"]
                            
                            # 타임스탬프 포맷 (분:초)
                            timestamp_str = time.strftime('%M:%S', time.gmtime(current_video_time))
                            
                            if duration < 1: # 계산대 0초 (안 들름)
                                logs.append({
                                    "timestamp": timestamp_str,
                                    "type": "위험 (계산대 미방문)",
                                    "videoUrl": "" # 필요시 클립 영상 주소 추가
                                })
                            elif duration < 20: # 20초 미만
                                logs.append({
                                    "timestamp": timestamp_str,
                                    "type": f"경고 (결제시간 부족: {int(duration)}초)",
                                    "videoUrl": ""
                                })
                            
                            customers[track_id]["status"] = "EXITED"

    cap.release()
    return logs