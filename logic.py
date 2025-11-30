import cv2
import time
from ultralytics import YOLO
from shapely.geometry import Point, Polygon

# 모델 로드
model = YOLO('yolov8n.pt')
# 좌표 설정
ZONE_CHECKOUT = Polygon([(820, 333), (1004, 339), (940, 718), (734, 714)])
ZONE_ENTRY = Polygon([(176, 181), (403, 193), (422, 636), (209, 691)])

def process_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    # [속도 개선 설정] 3프레임마다 1번씩만 분석 (3배 빨라짐)
    SKIP_FRAMES = 3 

    customers = {}
    logs = []
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # [속도 개선] 설정한 프레임 간격이 아니면 분석 건너뛰기
        if frame_count % SKIP_FRAMES != 0:
            continue
        
        # 현재 시간 (초)
        current_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # YOLO 추적
        results = model.track(frame, persist=True, classes=0, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                foot_point = Point((x1 + x2) / 2, y2)

                # 1. 입장
                if track_id not in customers:
                    if ZONE_ENTRY.contains(foot_point):
                        customers[track_id] = {
                            "entry_time": current_video_time,
                            "checkout_start": None,
                            "checkout_duration": 0,
                            "status": "SHOPPING"
                        }

                # 2. 계산대 체류
                if track_id in customers:
                    if ZONE_CHECKOUT.contains(foot_point):
                        if customers[track_id]["checkout_start"] is None:
                            customers[track_id]["checkout_start"] = current_video_time
                        # [시간 보정] 프레임을 건너뛰었으므로 시간은 (1/fps * 건너뛴 수) 만큼 더해줌
                        customers[track_id]["checkout_duration"] += (6.6 / fps) * SKIP_FRAMES
                        print(f"ID {track_id}: 계산대 체류 중... {customers[track_id]['checkout_duration']:.2f}초") 
                    else:
                        customers[track_id]["checkout_start"] = None

                    # 3. 퇴장 및 판단
                    if ZONE_ENTRY.contains(foot_point) and (current_video_time - customers[track_id]["entry_time"] > 5):
                        if customers[track_id]["status"] != "EXITED":
                            duration = customers[track_id]["checkout_duration"]
                            timestamp_str = time.strftime('%M:%S', time.gmtime(current_video_time))
                            
                            if duration < 1:
                                logs.append({
                                    "timestamp": timestamp_str,
                                    "type": "위험 (계산대 미방문)",
                                    "videoUrl": ""
                                })
                            elif duration < 20:
                                logs.append({
                                    "timestamp": timestamp_str,
                                    "type": f"경고 (결제시간 부족: {int(duration)}초)",
                                    "videoUrl": ""
                                })
                            
                            customers[track_id]["status"] = "EXITED"

    cap.release()
    return logs