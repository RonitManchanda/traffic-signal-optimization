import cv2
import math
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
VIDEO_PATH = "intersection.mp4"
MODEL_PATH = "yolo11s.pt"
TRACKER_CFG = "botsort_custom.yaml"

TARGET_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
BLOCKER_CLASSES = {2, 5, 7}       # bus, truck

CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45

MIN_BOX_AREA = 900
HISTORY_LEN = 14
MIN_TRACK_AGE = 5
MIN_HISTORY_FOR_DECISION = 6
MIN_DIRECTION_SCORE = 0.55
STRAIGHT_DIRECTION_SCORE = 0.88
MAX_MISSING_FRAMES = 40

# Turn logic
TURN_LANE_CONFIRM_FRAMES = 3
TURN_EXIT_DIRECTION_HITS = 2
TURN_EXIT_SCORE_THRESHOLD = 0.30

# Fallback / blocker logic
BLOCKER_MIN_BOX_AREA = 18000
STRAIGHT_BACKUP_SCORE_THRESHOLD = 0.50
STRAIGHT_BACKUP_LABELS = {"N", "NW", "NE"}

# =========================
# LANE POLYGON COORDINATES
# =========================
LANE_POLYGONS = {
    "south_to_west_turn": np.array([
        (1124, 885),
        (1256, 858),
        (1542, 1066),
        (1341, 1065)
    ], dtype=np.int32),

    "south_to_north_straight": np.array([
        (1432, 990),
        (1270, 862),
        (1551, 923),
        (1432, 990)
    ], dtype=np.int32),

    "south_to_east_turn": np.array([
        (1439, 846),
        (1593, 818),
        (1883, 966),
        (1786, 1059),
    ], dtype=np.int32),
    "east_to_south_turn": np.array([
        (1481, 651),
        (1420, 620),
        (1642, 604),
        (1652, 635),
    ], dtype=np.int32),
}

LANE_COLORS = {
    "south_to_west_turn": (0, 165, 255),      # orange
    "south_to_north_straight": (255, 255, 0), # yellow
    "south_to_east_turn": (0, 255, 255),
    "east_to_south_turn": (50, 200, 110),# cyan
}

# =========================
# X-Y AXIS COORDINATES
# =========================
Y_LINE = ((819, 550), (1258, 857))
X_LINE = ((180, 735), (1439, 621))

# =========================
# HELPER METHODS
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
    cv2.putText(frame, "Y_LINE", Y_LINE[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.line(frame, X_LINE[0], X_LINE[1], (255, 0, 255), 2)
    cv2.putText(frame, "X_LINE", X_LINE[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)


def motion_score(dx, dy, expected_vec):
    move_vec = normalize_vector(dx, dy)
    return dot(move_vec, expected_vec)


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
    return direction_label(p2[0] - p1[0], p2[1] - p1[1])


def recent_direction_label(history, recent_len=4):
    pts = list(history)
    if len(pts) < 2:
        return None

    pts = pts[-recent_len:] if len(pts) >= recent_len else pts
    if len(pts) < 2:
        return None

    return direction_label_from_points(pts[0], pts[-1])


def recent_motion_score(history, expected_vec, recent_len=4):
    pts = list(history)
    if len(pts) < 2:
        return None

    pts = pts[-recent_len:] if len(pts) >= recent_len else pts
    if len(pts) < 2:
        return None

    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    return motion_score(dx, dy, expected_vec)


def detect_lane(point):
    for lane_name, poly in LANE_POLYGONS.items():
        if point_in_polygon(point, poly):
            return lane_name
    return None


def cleanup_old_tracks(
    frame_index,
    track_history,
    track_age,
    lane_votes,
    committed_lane,
    last_seen,
    counted_track_ids,
    turn_lane_seen_frames,
    turn_exit_hits,
):
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
        turn_lane_seen_frames.pop(track_id, None)
        turn_exit_hits.pop(track_id, None)
        # counted_track_ids intentionally kept


# =========================
# AXIS VECTOR CALIBRATION
# =========================
Y_AXIS = vector_from_points(*Y_LINE)
X_AXIS = vector_from_points(*X_LINE)

EXPECTED_MOTION = {
    "south_to_north_straight": (-Y_AXIS[0], -Y_AXIS[1]),
}

# Both turn lanes are true turn lanes now
TURN_EXPECTED_MOTION = {
    "south_to_west_turn": {
        "exit_vec": (-X_AXIS[0], -X_AXIS[1]),
        # tune these to what you actually see on screen
        "exit_labels": {"W"},
    },
    "south_to_east_turn": {
        "exit_vec": (X_AXIS[0], X_AXIS[1]),
        "exit_labels": {"E", "NE"},
    },
    "east_to_south_turn": {
        "exit_vec": (X_AXIS[0], -X_AXIS[1]),
        "exit_labels": {"S", "SE", "SW"},
    },
}

# =========================
# MAIN
# =========================
def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    counts = {
        "south_to_west_turn": 0,
        "south_to_north_straight": 0,
        "south_to_east_turn": 0,
        "east_to_south_turn": 0,
    }

    event_log = []

    counted_track_ids = set()
    track_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    track_age = defaultdict(int)
    lane_votes = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    committed_lane = {}
    last_seen = {}

    # new state
    turn_lane_seen_frames = defaultdict(int)
    turn_exit_hits = defaultdict(int)

    cv2.namedWindow("Lane Test", cv2.WINDOW_NORMAL)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_seconds = frame_index / fps if fps > 0 else 0.0

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

        # per-frame blocker flags
        blocker_active = {
            "south_to_west_turn": False,
            "south_to_east_turn": False,
        }

        if result.boxes is not None:
            # pass 1: detect blockers
            for box in result.boxes:
                if box.id is None:
                    continue

                cls_id = int(box.cls[0].item())
                if cls_id not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                box_area = (x2 - x1) * (y2 - y1)
                if box_area < MIN_BOX_AREA:
                    continue

                road_point = get_bottom_center(x1, y1, x2, y2)
                lane_now = detect_lane(road_point)

                if (
                    cls_id in BLOCKER_CLASSES
                    and box_area >= BLOCKER_MIN_BOX_AREA
                    and lane_now in blocker_active
                ):
                    blocker_active[lane_now] = True

            # pass 2: normal processing
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

                history = track_history[track_id]

                # label
                label = f"ID {track_id}"
                if track_id in committed_lane:
                    label += f" | {committed_lane[track_id]}"

                if len(history) >= 2:
                    dir_label = recent_direction_label(history, recent_len=4)
                    if dir_label is None:
                        dir_label = "..."
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

                if len(history) >= 2:
                    draw_motion_arrow(frame, history[0], history[-1], (120, 255, 120))

                if track_id in counted_track_ids:
                    continue

                if track_age[track_id] < MIN_TRACK_AGE:
                    continue

                if len(history) < MIN_HISTORY_FOR_DECISION:
                    continue

                dx = history[-1][0] - history[0][0]
                dy = history[-1][1] - history[0][1]
                full_dir_label = direction_label(dx, dy)
                nx, ny = normalize_vector(dx, dy)

                recent_dir = recent_direction_label(history, recent_len=4)
                lane_name = committed_lane.get(track_id)
                straight_score = motion_score(dx, dy, EXPECTED_MOTION["south_to_north_straight"])

                # =========================
                # TURN LANES
                # =========================
                if lane_name in TURN_EXPECTED_MOTION:
                    turn_lane_seen_frames[track_id] += 1

                    exit_vec = TURN_EXPECTED_MOTION[lane_name]["exit_vec"]
                    exit_labels = TURN_EXPECTED_MOTION[lane_name]["exit_labels"]

                    current_exit_score = recent_motion_score(history, exit_vec, recent_len=4)
                    if current_exit_score is None:
                        continue

                    in_exit_direction = (
                        recent_dir in exit_labels
                        and current_exit_score >= TURN_EXIT_SCORE_THRESHOLD
                    )

                    if in_exit_direction:
                        turn_exit_hits[track_id] += 1
                    else:
                        turn_exit_hits[track_id] = 0

                    lane_ready = turn_lane_seen_frames[track_id] >= TURN_LANE_CONFIRM_FRAMES
                    exit_ready = turn_exit_hits[track_id] >= TURN_EXIT_DIRECTION_HITS

                    cv2.putText(
                        frame,
                        f"turn_seen={turn_lane_seen_frames[track_id]}",
                        (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

                    cv2.putText(
                        frame,
                        f"recent={recent_dir} exit_score={current_exit_score:.2f}",
                        (x1, y2 + 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

                    cv2.putText(
                        frame,
                        f"lane_ready={lane_ready} exit_hits={turn_exit_hits[track_id]}",
                        (x1, y2 + 54),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0) if exit_ready else (0, 0, 255),
                        1
                    )

                    cv2.putText(
                        frame,
                        f"vec=({nx:.2f},{ny:.2f})",
                        (x1, y2 + 72),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1
                    )

                    # only count once the vehicle actually changes into exit direction
                    if lane_ready and exit_ready:
                        counts[lane_name] += 1
                        counted_track_ids.add(track_id)

                        timestamp_str = f"{timestamp_seconds:.2f}s"
                        event = {
                            "time": timestamp_str,
                            "track_id": track_id,
                            "lane": lane_name,
                            "type": "TURN",
                            "direction": recent_dir,
                            "exit_score": round(current_exit_score, 2),
                            "vec": (round(nx, 2), round(ny, 2)),
                        }
                        event_log.append(event)

                        print(
                            f"[{timestamp_str}] TURN | ID {track_id} | "
                            f"{lane_name} | dir={recent_dir} | exit_score={current_exit_score:.2f}"
                        )

                        cv2.putText(
                            frame,
                            "COUNTED TURN",
                            (road_point[0] + 10, road_point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            LANE_COLORS[lane_name],
                            2
                        )

                    continue

                # =========================
                # STRAIGHT LANE / BACKUP
                # =========================
                cv2.putText(
                    frame,
                    f"straight_score={straight_score:.2f}",
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

                # normal straight count
                normal_straight_ok = (
                    lane_name == "south_to_north_straight"
                    and straight_score >= STRAIGHT_DIRECTION_SCORE
                )

                # blocker fallback:
                # if a big truck/bus is in either turn lane and this visible track is moving straight,
                # let it count as straight even if lane commitment is missing/noisy.
                fallback_straight_ok = (
                    (
                        blocker_active["south_to_west_turn"]
                        or blocker_active["south_to_east_turn"]
                    )
                    and lane_now == "south_to_north_straight"
                    and recent_dir in STRAIGHT_BACKUP_LABELS
                    and straight_score >= STRAIGHT_BACKUP_SCORE_THRESHOLD
                )

                if normal_straight_ok or fallback_straight_ok:
                    counts["south_to_north_straight"] += 1
                    counted_track_ids.add(track_id)

                    timestamp_str = f"{timestamp_seconds:.2f}s"
                    event = {
                        "time": timestamp_str,
                        "track_id": track_id,
                        "lane": "south_to_north_straight",
                        "type": "STRAIGHT_BACKUP" if fallback_straight_ok and not normal_straight_ok else "STRAIGHT",
                        "direction": full_dir_label,
                        "score": round(straight_score, 2),
                        "vec": (round(nx, 2), round(ny, 2)),
                    }
                    event_log.append(event)

                    print(
                        f"[{timestamp_str}] {event['type']} | ID {track_id} | "
                        f"south_to_north_straight | {full_dir_label} | score={straight_score:.2f}"
                    )

                    cv2.putText(
                        frame,
                        f"COUNTED {event['type']}",
                        (road_point[0] + 10, road_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        LANE_COLORS["south_to_north_straight"],
                        2
                    )

        cleanup_old_tracks(
            frame_index,
            track_history,
            track_age,
            lane_votes,
            committed_lane,
            last_seen,
            counted_track_ids,
            turn_lane_seen_frames,
            turn_exit_hits,
        )

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

        # blocker display
        by = y + 10
        for lane_name, active in blocker_active.items():
            cv2.putText(
                frame,
                f"blocker_{lane_name}: {active}",
                (20, by),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255) if active else (180, 180, 180),
                2
            )
            by += 28

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

    print("\n========== EVENT LOG ==========")
    for event in event_log:
        print(event)
    print("================================")


if __name__ == "__main__":
    main()