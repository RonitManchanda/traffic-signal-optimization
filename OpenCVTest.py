import cv2
import math
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
VIDEO_PATH = "intersection.mp4"
MODEL_PATH = "yolov8m.pt"
TRACKER_CFG = "botsort.yaml"

TARGET_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.50

MIN_BOX_AREA = 900
HISTORY_LEN = 12
MIN_TRACK_AGE = 5
MIN_HISTORY_FOR_DECISION = 6
MIN_DIRECTION_SCORE = 0.55
MAX_MISSING_FRAMES = 40

# turn-specific thresholds
TURN_APPROACH_THRESHOLD = 0.20
TURN_EXIT_THRESHOLD = 0.35

# =========================
# USER-PROVIDED LANE POLYGONS
# =========================
LANE_POLYGONS = {
    "south_to_west_turn": np.array([
        (1124, 885),
        (1256, 858),
        (1542, 1066),
        (1341, 1065)
    ], dtype=np.int32),

    "south_to_north_straight": np.array([
        (1414, 836),
        (1264, 855),
        (1550, 1061),
        (1778, 1062)
    ], dtype=np.int32),

    "south_to_east_turn": np.array([
        (1439, 846),
        (1593, 818),
        (1883, 966),
        (1786, 1059),
    ], dtype=np.int32),
}

LANE_COLORS = {
    "south_to_west_turn": (0, 165, 255),      # orange
    "south_to_north_straight": (255, 255, 0), # yellow
    "south_to_east_turn": (0, 255, 255),      # cyan/yellow-ish
}

# =========================
# USER-PROVIDED AXIS LINES
# =========================
Y_LINE = ((819, 550), (1258, 857))
X_LINE = ((180, 735), (1439, 621))

# =========================
# HELPERS
# =========================
def normalize_vector(dx, dy):
    mag = math.sqrt(dx * dx + dy * dy)
    if mag == 0:
        return (0.0, 0.0)
    return (dx / mag, dy / mag)


def vector_from_points(p1, p2):
    return normalize_vector(p2[0] - p1[0], p2[1] - p1[1])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def get_bottom_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, y2)


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def draw_polygons(frame):
    for lane_name, poly in LANE_POLYGONS.items():
        color = LANE_COLORS[lane_name]
        cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=3)

        x, y = poly[0]
        cv2.putText(
            frame,
            lane_name,
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )


def draw_axis_lines(frame):
    cv2.line(frame, Y_LINE[0], Y_LINE[1], (0, 255, 0), 2)
    cv2.putText(
        frame,
        "Y_LINE",
        Y_LINE[0],
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.line(frame, X_LINE[0], X_LINE[1], (255, 0, 255), 2)
    cv2.putText(
        frame,
        "X_LINE",
        X_LINE[0],
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2
    )


def motion_score(dx, dy, expected_vec):
    move_vec = normalize_vector(dx, dy)
    return dot(move_vec, expected_vec)


def segmented_turn_scores(history, approach_vec, exit_vec):
    if len(history) < 6:
        return None, None, None, None

    pts = list(history)
    mid = len(pts) // 2

    first_half = pts[:mid]
    second_half = pts[mid:]

    if len(first_half) < 2 or len(second_half) < 2:
        return None, None, None, None

    dx1 = first_half[-1][0] - first_half[0][0]
    dy1 = first_half[-1][1] - first_half[0][1]

    dx2 = second_half[-1][0] - second_half[0][0]
    dy2 = second_half[-1][1] - second_half[0][1]

    approach_score = motion_score(dx1, dy1, approach_vec)
    exit_score = motion_score(dx2, dy2, exit_vec)

    return approach_score, exit_score, first_half, second_half


def draw_motion_arrow(frame, start_pt, end_pt, color):
    cv2.arrowedLine(frame, start_pt, end_pt, color, 2, tipLength=0.25)


def direction_label(dx, dy):
    nx, ny = normalize_vector(dx, dy)

    if abs(nx) < 0.2 and abs(ny) < 0.2:
        return "STILL"

    horizontal = ""
    vertical = ""

    if nx > 0.3:
        horizontal = "E"
    elif nx < -0.3:
        horizontal = "W"

    if ny > 0.3:
        vertical = "S"   # down in image
    elif ny < -0.3:
        vertical = "N"   # up in image

    return vertical + horizontal if (vertical + horizontal) != "" else "?"


def direction_label_from_points(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return direction_label(dx, dy)


def segmented_turn_direction_labels(history):
    if len(history) < 6:
        return None, None

    pts = list(history)
    mid = len(pts) // 2

    first_half = pts[:mid]
    second_half = pts[mid:]

    if len(first_half) < 2 or len(second_half) < 2:
        return None, None

    approach_label = direction_label_from_points(first_half[0], first_half[-1])
    exit_label = direction_label_from_points(second_half[0], second_half[-1])

    return approach_label, exit_label


def detect_lane(point):
    for lane_name, poly in LANE_POLYGONS.items():
        if point_in_polygon(point, poly):
            return lane_name
    return None


def cleanup_old_tracks(frame_index, track_history, track_age, lane_votes, committed_lane, last_seen, counted_track_ids):
    stale_ids = [
        track_id for track_id, seen_frame in last_seen.items()
        if frame_index - seen_frame > MAX_MISSING_FRAMES
    ]

    for track_id in stale_ids:
        track_history.pop(track_id, None)
        track_age.pop(track_id, None)
        lane_votes.pop(track_id, None)
        committed_lane.pop(track_id, None)
        last_seen.pop(track_id, None)
        # counted_track_ids intentionally kept so old IDs do not get counted again


# =========================
# AXIS CALIBRATION
# =========================
Y_AXIS = vector_from_points(*Y_LINE)
X_AXIS = vector_from_points(*X_LINE)

# Straight-lane expected motion.
# If labels look backward, flip the signs here.
EXPECTED_MOTION = {
    "south_to_north_straight": (-Y_AXIS[0], -Y_AXIS[1]),
    "south_to_east_turn": (X_AXIS[0], X_AXIS[1]),
}

# Turn lane uses phase-based checking:
# 1. approach along Y-ish direction
# 2. exit along westbound X-ish direction
TURN_EXPECTED_MOTION = {
    "south_to_west_turn": {
        "approach": (-Y_AXIS[0], -Y_AXIS[1]),
        "exit": (-X_AXIS[0], -X_AXIS[1])
    }

}

# Compass-label transition check for turn lanes.
# Start loose, then tighten after you inspect real outputs.
TURN_DIRECTION_LABELS = {
    "south_to_west_turn": {
        "approach": {"NW", "W"},
        "exit": {"W"}
    },
    "south_to_east_turn": {
        "approach": {"NW", "W"},
        "exit": {"E"}
    }


}



# =========================
# MAIN
# =========================
def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    counts = {
        "south_to_west_turn": 0,
        "south_to_north_straight": 0,
        "south_to_east_turn": 0,
    }

    counted_track_ids = set()
    track_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    track_age = defaultdict(int)
    lane_votes = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    committed_lane = {}
    last_seen = {}

    cv2.namedWindow("Lane Test", cv2.WINDOW_NORMAL)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw_polygons(frame)
        draw_axis_lines(frame)

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CFG,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
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

                box_area = (x2 - x1) * (y2 - y1)
                if box_area < MIN_BOX_AREA:
                    continue

                road_point = get_bottom_center(x1, y1, x2, y2)

                last_seen[track_id] = frame_index
                track_age[track_id] += 1
                track_history[track_id].append(road_point)

                # draw detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.circle(frame, road_point, 4, (0, 0, 255), -1)

                lane_now = detect_lane(road_point)
                if lane_now is not None:
                    lane_votes[track_id].append(lane_now)

                # commit lane after repeated agreement
                if track_id not in committed_lane and len(lane_votes[track_id]) >= 4:
                    votes = list(lane_votes[track_id])
                    for lane_name in set(votes):
                        if votes.count(lane_name) >= 4:
                            committed_lane[track_id] = lane_name
                            break

                # build label safely
                label = f"ID {track_id}"

                if track_id in committed_lane:
                    label += f" | {committed_lane[track_id]}"

                history = track_history[track_id]
                if len(history) >= 2:
                    dx_label = history[-1][0] - history[0][0]
                    dy_label = history[-1][1] - history[0][1]
                    dir_label = direction_label(dx_label, dy_label)
                    label += f" | {dir_label}"
                else:
                    label += " | ..."

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                if track_id in counted_track_ids:
                    continue

                if track_age[track_id] < MIN_TRACK_AGE:
                    continue

                if track_id not in committed_lane:
                    continue

                history = track_history[track_id]
                if len(history) < MIN_HISTORY_FOR_DECISION:
                    continue

                dx = history[-1][0] - history[0][0]
                dy = history[-1][1] - history[0][1]
                nx, ny = normalize_vector(dx, dy)

                lane_name = committed_lane[track_id]

                if lane_name == "south_to_west_turn":
                    approach_vec = TURN_EXPECTED_MOTION[lane_name]["approach"]
                    exit_vec = TURN_EXPECTED_MOTION[lane_name]["exit"]

                    approach_score, exit_score, first_half, second_half = segmented_turn_scores(
                        history,
                        approach_vec,
                        exit_vec
                    )

                    if approach_score is None or exit_score is None:
                        continue

                    approach_ok = approach_score >= TURN_APPROACH_THRESHOLD
                    exit_ok = exit_score >= TURN_EXIT_THRESHOLD

                    approach_label, exit_label = segmented_turn_direction_labels(history)
                    if approach_label is None or exit_label is None:
                        continue

                    allowed_approach = TURN_DIRECTION_LABELS[lane_name]["approach"]
                    allowed_exit = TURN_DIRECTION_LABELS[lane_name]["exit"]

                    approach_label_ok = approach_label in allowed_approach
                    exit_label_ok = exit_label in allowed_exit

                    cv2.putText(
                        frame,
                        f"approach={approach_score:.2f}",
                        (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

                    cv2.putText(
                        frame,
                        f"exit={exit_score:.2f}",
                        (x1, y2 + 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

                    cv2.putText(
                        frame,
                        f"A:{approach_ok} E:{exit_ok}",
                        (x1, y2 + 54),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0) if (approach_ok and exit_ok) else (0, 0, 255),
                        1
                    )

                    cv2.putText(
                        frame,
                        f"{approach_label}->{exit_label}",
                        (x1, y2 + 72),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0) if (approach_label_ok and exit_label_ok) else (0, 0, 255),
                        1
                    )

                    cv2.putText(
                        frame,
                        f"vec=({nx:.2f},{ny:.2f})",
                        (x1, y2 + 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1
                    )

                    if first_half is not None and len(first_half) >= 2:
                        draw_motion_arrow(frame, first_half[0], first_half[-1], (0, 255, 255))
                    if second_half is not None and len(second_half) >= 2:
                        draw_motion_arrow(frame, second_half[0], second_half[-1], (255, 0, 0))

                    if approach_ok and exit_ok and approach_label_ok and exit_label_ok:
                        counts[lane_name] += 1
                        counted_track_ids.add(track_id)

                        cv2.putText(
                            frame,
                            "COUNTED TURN",
                            (road_point[0] + 10, road_point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            LANE_COLORS[lane_name],
                            2
                        )

                else:
                    expected_vec = EXPECTED_MOTION[lane_name]
                    score = motion_score(dx, dy, expected_vec)

                    cv2.putText(
                        frame,
                        f"score={score:.2f}",
                        (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

                    cv2.putText(
                        frame,
                        f"vec=({nx:.2f},{ny:.2f})",
                        (x1, y2 + 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1
                    )

                    if score >= MIN_DIRECTION_SCORE:
                        counts[lane_name] += 1
                        counted_track_ids.add(track_id)

                        cv2.putText(
                            frame,
                            f"COUNTED {lane_name}",
                            (road_point[0] + 10, road_point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            LANE_COLORS[lane_name],
                            2
                        )

        cleanup_old_tracks(
            frame_index,
            track_history,
            track_age,
            lane_votes,
            committed_lane,
            last_seen,
            counted_track_ids
        )

        # display counts
        y = 35
        for name, count in counts.items():
            cv2.putText(
                frame,
                f"{name}: {count}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                LANE_COLORS[name],
                2
            )
            y += 35

        cv2.imshow("Lane Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\n========== FINAL COUNTS ==========")
    for lane, count in counts.items():
        print(f"{lane}: {count}")
    print("==================================")


if __name__ == "__main__":
    main()