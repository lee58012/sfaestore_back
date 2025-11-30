import cv2
import numpy as np

# 분석할 영상 파일 이름 (여기를 내 영상 파일명으로 바꾸세요!)
VIDEO_PATH = "temp_videos/test1.mp4" 

# 좌표를 저장할 리스트
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"좌표 클릭: ({x}, {y})")
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1) # 빨간 점 표시
        cv2.imshow("Image", img)

# 영상 열기
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()

if not ret:
    print("영상을 찾을 수 없습니다. 경로를 확인하세요.")
    exit()

img = frame
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)

print("\n=== [1단계: 계산대(Checkout) 구역 설정] ===")
print("계산대 구역의 모서리 4군데를 순서대로 클릭하세요.")
print("다 찍었으면 아무 키나 누르세요.")

cv2.waitKey(0)

print(f"\n✅ 계산대 좌표 복사해서 쓰세요:\nZONE_CHECKOUT = Polygon({points})")

# 초기화 후 입구 설정
points = []
img = frame.copy() # 화면 초기화
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)

print("\n=== [2단계: 입구(Entry) 구역 설정] ===")
print("입구 구역의 모서리 4군데를 순서대로 클릭하세요.")
print("다 찍었으면 아무 키나 누르세요.")

cv2.waitKey(0)
print(f"\n✅ 입구 좌표 복사해서 쓰세요:\nZONE_ENTRY = Polygon({points})")

cv2.destroyAllWindows()
cap.release()