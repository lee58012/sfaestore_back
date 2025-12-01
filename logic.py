import cv2
import time
from shapely.geometry import Point, Polygon, box

def get_model():
    global model
    if model is None:
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            print("YOLO 모델 로드 완료")
        except Exception as e:
            print("YOLO 모델 로드 실패:", e)
            raise
    return model

model = None
ZONE_CHECKOUT = Polygon([(468, 120), (672, 110), (661, 469), (487, 472)])
ZONE_ENTRY = Polygon([(112, 131), (265, 136), (281, 439), (136, 471)])
MIN_STORE_STAY = 5

def process_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    time_per_frame = 1.0 / fps
    model = get_model()

    frame_count = 0
    all_checkout_frames = []
    last_person_frame = None

    # 단일 루프: 프레임을 한 번만 읽음(무한루프 방지)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # detect (ID 추적 없이 계산대 체류 감지)
        results = model(frame, verbose=False, classes=0)
        # 전체 프레임에서 사람 감지가 있으면 마지막 감지 프레임 업데이트
        if results and len(results[0].boxes) > 0:
            last_person_frame = frame_count
        if results[0].boxes.xyxy is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu()
            for box_coords in boxes:
                x1, y1, x2, y2 = box_coords
                # 박스-폴리곤 교차로 판정: 교차 면적 비율이 충분히 큰 경우에만 계산대 체류로 간주
                box_poly = box(x1, y1, x2, y2)
                if not ZONE_CHECKOUT.intersects(box_poly):
                    continue
                inter = ZONE_CHECKOUT.intersection(box_poly)
                if inter.is_empty:
                    continue
                overlap_ratio = inter.area / box_poly.area
                # 임계값: 교차 면적 / 박스 면적
                # 0.12로 설정하면 passing-by를 대부분 걸러내면서 짧은 체류(≈1s)를 일부 포착합니다.
                MIN_OVERLAP_RATIO = 0.12
                if overlap_ratio >= MIN_OVERLAP_RATIO:
                    all_checkout_frames.append(frame_count)
                    break

    cap.release()

    logs = []
    if not all_checkout_frames:
        return logs

    # 연속 구간 분리
    segments = []
    seg_start = all_checkout_frames[0]
    prev = seg_start
    MAX_ABSENCE_FRAMES = int(5.0 / time_per_frame)
    for f in all_checkout_frames[1:]:
        if f - prev > MAX_ABSENCE_FRAMES:
            segments.append((seg_start, prev))
            seg_start = f
        prev = f
    segments.append((seg_start, prev))

    # 가장 긴 구간 선택
    best = max(segments, key=lambda s: s[1] - s[0])
    duration = (best[1] - best[0]) * time_per_frame

    if duration >= 15:
        # 정상: 로그를 반환하지 않음 (영상 저장 X)
        return logs
    elif duration >= 3:
        log_type = f"경고 (결제시간 부족: {int(duration)}초)"
    else:
        log_type = f"위험 (계산대 미방문: {int(duration)}초)"

    # 이상행동(경고/위험)인 경우: 타임스탬프 결정
    # - 경고: 계산대에서의 마지막 감지 프레임(최장 구간의 끝)
    # - 위험(계산대 미방문): 나가기 직전의 영상 -> 마지막으로 사람 감지된 프레임 사용
    if log_type.startswith("경고"):
        timestamp_frame = best[1]
    else:
        # 위험인 경우: 전체 프레임에서 마지막으로 사람 감지된 프레임을 우선 사용
        timestamp_frame = last_person_frame if last_person_frame is not None else all_checkout_frames[-1]

    timestamp_str = time.strftime('%M:%S', time.gmtime(timestamp_frame * time_per_frame))
    logs.append({"timestamp": timestamp_str, "type": log_type, "videoUrl": ""})
    return logs