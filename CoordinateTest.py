import cv2

VIDEO_PATH = "intersection.mp4"

points = []
paused = True

def click_event(event, x, y, flags, param):
    global frame_display

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"{x}, {y}")
        points.append((x, y))

        # draw point
        cv2.circle(frame_display, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Video", frame_display)


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video")

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not read frame")

frame_display = frame.copy()

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Video", click_event)

print("Press SPACE to pause/play, ESC to exit")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame_display = frame.copy()

    cv2.imshow("Video", frame_display)

    key = cv2.waitKey(30) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        paused = not paused

cap.release()
cv2.destroyAllWindows()

print("Collected points:", points)