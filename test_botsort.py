import cv2
import math
import csv
from collections import defaultdict, deque
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
VIDEO_PATH = "intersection.mp4"
MODEL_PATH = "yolov8m.pt"
TRACKER_CFG = "botsort.yaml"

TARGET_CLASSES = {2, 3, 5, 7}

COUNT_LINES = {
    "northbound": ((1128, 882), (1595, 815)),
    "southbound": ((814, 557), (437, 594)),
    "eastbound":  ((244, 728), (312, 874)),
    "westbound":  ((1321, 570), (1478, 649)),
}

COLORS = {
    "northbound": (255, 0, 0),
    "southbound": (0, 255, 0),
    "eastbound":  (255, 0, 255),
    "westbound":  (0, 255, 255),
}

MIN_TRACK_AGE = 5
HISTORY_LEN = 8
MAX_LINE_DISTANCE = 40
MIN_DIRECTION_SCORE = 0.65
OUTPUT_VIDEO = "botsort_output.mp4"
OUTPUT_LOG = "botsort_events.csv"


def get_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def point_side_of_line(point, line):
    (x1, y1), (x2, y2) = line
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def crossed_line(prev_pt, curr_pt, line):
    prev_side = point_side_of_line(prev_pt, line)
    curr_side = point_side_of_line(curr_pt, line)

    if prev_side == 0 or curr_side == 0:
        return False

    return (prev_side < 0 < curr_side) or (prev_side > 0 > curr_side)


def point_line_distance(point, line):
    (x1, y1), (x2, y2) = line
    px, py = point
    numerator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return 999999 if denominator == 0 else numerator / denominator


def unit_vector(dx, dy):
    mag = math.sqrt(dx * dx + dy * dy)
    if mag == 0:
        return (0.0, 0.0)
    return (dx / mag, dy / mag)


def direction_vector(direction):
    if direction == "northbound":
        return (0.0, -1.0)
    if direction == "southbound":
        return (0.0, 1.0)
    if direction == "eastbound":
        return (1.0, 0.0)
    if direction == "westbound":
        return (-1.0, 0.0)
    return (0.0, 0.0)


def motion_score(dx, dy, direction):
    ux, uy = unit_vector(dx, dy)
    tx, ty = direction_vector(direction)
    return ux * tx + uy * ty


def draw_count_lines(frame):
    for name, (p1, p2) in COUNT_LINES.items():
        color = COLORS[name]
        cv2.line(frame, p1, p2, color, 4)
        cv2.circle(frame, p1, 5, color, -1)
        cv2.circle(frame, p2, 5, color, -1)
        cv2.putText(
            frame,
            name,
            (min(p1[0], p2[0]), max(20, min(p1[1], p2[1]) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    counts = {name: 0 for name in COUNT_LINES}
    counted_track_ids = set()
    track_age = defaultdict(int)
    track_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))

    with open(OUTPUT_LOG, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["frame", "track_id", "direction", "x", "y"])

        frame_idx = 0
        cv2.namedWindow("BoT-SORT Test", cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            draw_count_lines(frame)

            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER_CFG,
                conf=0.20,
                iou=0.50,
                verbose=False
            )

            result = results[0]

            if result.boxes is not None:
                for box in result.boxes:
                    if box.id is None:
                        continue

                    cls_id = int(box.cls[0].item())
                    if cls_id not in TARGET_CLASSES:
                        continue

                    track_id = int(box.id[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    center = get_center(x1, y1, x2, y2)

                    track_age[track_id] += 1
                    track_history[track_id].append(center)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                    cv2.circle(frame, center, 4, (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        f"ID {track_id}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )

                    if track_id in counted_track_ids:
                        continue
                    if track_age[track_id] < MIN_TRACK_AGE:
                        continue
                    if len(track_history[track_id]) < 4:
                        continue

                    history = list(track_history[track_id])
                    prev_pt = history[-2]
                    curr_pt = history[-1]
                    dx = history[-1][0] - history[0][0]
                    dy = history[-1][1] - history[0][1]

                    candidates = []

                    for direction, line in COUNT_LINES.items():
                        if not crossed_line(prev_pt, curr_pt, line):
                            continue

                        d_prev = point_line_distance(prev_pt, line)
                        d_curr = point_line_distance(curr_pt, line)
                        nearest_dist = min(d_prev, d_curr)

                        if nearest_dist > MAX_LINE_DISTANCE:
                            continue

                        score = motion_score(dx, dy, direction)
                        if score < MIN_DIRECTION_SCORE:
                            continue

                        candidates.append((direction, score, nearest_dist))

                    if candidates:
                        candidates.sort(key=lambda x: (-x[1], x[2]))
                        chosen_direction = candidates[0][0]

                        counts[chosen_direction] += 1
                        counted_track_ids.add(track_id)

                        csv_writer.writerow([
                            frame_idx,
                            track_id,
                            chosen_direction,
                            center[0],
                            center[1]
                        ])

                        cv2.putText(
                            frame,
                            f"COUNTED {chosen_direction}",
                            (center[0] + 10, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            COLORS[chosen_direction],
                            2
                        )

            y = 35
            for name, count in counts.items():
                cv2.putText(
                    frame,
                    f"{name}: {count}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    COLORS[name],
                    2
                )
                y += 35

            writer.write(frame)
            cv2.imshow("BoT-SORT Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("Done.")
    print(f"Saved video: {OUTPUT_VIDEO}")
    print(f"Saved events: {OUTPUT_LOG}")


if __name__ == "__main__":
    main()