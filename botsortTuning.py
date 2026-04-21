import os
import csv
import math
from collections import defaultdict, deque
from itertools import product

import cv2
from ultralytics import YOLO


# =========================
# USER CONFIG
# =========================
VIDEO_PATH = "intersection.mp4"
MODEL_PATH = "yolo11s.pt"

# Trackers
TRACKER_CONFIGS = [
    "botsort_custom.yaml",
    "bytetrack.yaml",
]

CONF_VALUES = [0.15, 0.20]
IOU_VALUES = [0.45, 0.50]

# Vehicle classes only
TARGET_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# Ignore tiny detections
MIN_BOX_AREA = 900

# How many frames a track can disappear before we consider it "gone"
MAX_GAP_FOR_RETURN = 20

# Heuristic thresholds for probable re-ID detection
RETURN_DISTANCE_THRESHOLD = 90
CENTER_MATCH_DISTANCE = 70

# Short tracks usually indicate unstable IDs
SHORT_TRACK_MAX_LEN = 5

# Output
SUMMARY_CSV = "tracker_tuning_summary.csv"
DETAILS_DIR = "tracker_tuning_details"


# =========================
# HELPERS
# =========================
def box_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def safe_name(text):
    return (
        text.replace(".yaml", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )


# =========================
# CORE ANALYSIS
# =========================
def analyze_run(video_path, model_path, tracker_cfg, conf_thresh, iou_thresh):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_index = 0

    # Per-track history
    track_frames = defaultdict(list)
    track_centers = defaultdict(list)
    track_classes = {}
    last_seen = {}
    last_center = {}

    total_vehicle_boxes = 0

    # For heuristics
    probable_reid_events = 0
    return_events = 0

    # Tracks that disappeared recently
    recently_lost = deque()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker=tracker_cfg,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False
        )

        result = results[0]

        current_tracks = []

        if result.boxes is not None:
            for box in result.boxes:
                if box.id is None:
                    continue

                cls_id = int(box.cls[0].item())
                if cls_id not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                if area < MIN_BOX_AREA:
                    continue

                track_id = int(box.id[0].item())
                center = box_center(x1, y1, x2, y2)

                total_vehicle_boxes += 1
                current_tracks.append((track_id, cls_id, center))

                track_frames[track_id].append(frame_index)
                track_centers[track_id].append(center)
                track_classes[track_id] = cls_id

                # Check whether this new/returning track looks like a recent lost one
                if track_id not in last_seen:
                    best_match = None
                    best_dist = float("inf")

                    for lost in list(recently_lost):
                        lost_id, lost_cls, lost_frame, lost_center_pt = lost

                        if cls_id != lost_cls:
                            continue

                        gap = frame_index - lost_frame
                        if gap < 1 or gap > MAX_GAP_FOR_RETURN:
                            continue

                        dist = euclidean(center, lost_center_pt)
                        if dist < best_dist:
                            best_dist = dist
                            best_match = lost

                    if best_match is not None and best_dist <= RETURN_DISTANCE_THRESHOLD:
                        probable_reid_events += 1
                        return_events += 1

                last_seen[track_id] = frame_index
                last_center[track_id] = center

        # Update recently lost list
        active_ids = {tid for tid, _, _ in current_tracks}

        # Add newly lost tracks
        for tid, seen_frame in list(last_seen.items()):
            if tid in active_ids:
                continue

            if seen_frame == frame_index - 1:
                recently_lost.append(
                    (tid, track_classes.get(tid), seen_frame, last_center.get(tid))
                )

        # Trim old lost tracks
        while recently_lost and frame_index - recently_lost[0][2] > MAX_GAP_FOR_RETURN:
            recently_lost.popleft()

        # Same-frame nearby same-class different-ID heuristic
        # If many similar boxes are close together, ID churn may be happening
        for i in range(len(current_tracks)):
            id1, cls1, c1 = current_tracks[i]
            for j in range(i + 1, len(current_tracks)):
                id2, cls2, c2 = current_tracks[j]
                if cls1 != cls2:
                    continue
                if euclidean(c1, c2) < CENTER_MATCH_DISTANCE:
                    # nearby same-class tracks can be normal in traffic,
                    # so this is only a soft indicator, not a true error count
                    pass

        frame_index += 1

    cap.release()

    # Summaries
    total_tracks = len(track_frames)
    track_lengths = {tid: len(frames) for tid, frames in track_frames.items()}

    short_tracks = sum(1 for length in track_lengths.values() if length <= SHORT_TRACK_MAX_LEN)
    avg_track_length = (
        sum(track_lengths.values()) / total_tracks if total_tracks > 0 else 0.0
    )
    max_track_length = max(track_lengths.values()) if total_tracks > 0 else 0
    min_track_length = min(track_lengths.values()) if total_tracks > 0 else 0

    fragmented_tracks = 0
    total_gaps = 0

    for tid, frames in track_frames.items():
        gaps = 0
        for k in range(1, len(frames)):
            if frames[k] - frames[k - 1] > 1:
                gaps += 1
        if gaps > 0:
            fragmented_tracks += 1
            total_gaps += gaps

    score = {
        "tracker": tracker_cfg,
        "conf": conf_thresh,
        "iou": iou_thresh,
        "frames_processed": frame_index,
        "total_vehicle_boxes": total_vehicle_boxes,
        "total_unique_track_ids": total_tracks,
        "short_tracks": short_tracks,
        "avg_track_length": round(avg_track_length, 2),
        "min_track_length": min_track_length,
        "max_track_length": max_track_length,
        "fragmented_tracks": fragmented_tracks,
        "total_internal_gaps": total_gaps,
        "probable_reid_events": probable_reid_events,
        "return_events": return_events,
    }

    details = {
        "track_lengths": track_lengths,
        "track_frames": track_frames,
        "track_classes": track_classes,
    }

    return score, details


def write_details(details_path, details):
    with open(details_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "class_id", "track_length", "first_frame", "last_frame"])

        for tid in sorted(details["track_lengths"].keys()):
            frames = details["track_frames"][tid]
            writer.writerow([
                tid,
                details["track_classes"].get(tid, ""),
                details["track_lengths"][tid],
                frames[0],
                frames[-1],
            ])


def main():
    os.makedirs(DETAILS_DIR, exist_ok=True)

    all_rows = []

    combos = list(product(TRACKER_CONFIGS, CONF_VALUES, IOU_VALUES))
    print(f"Running {len(combos)} tracker tests on the same video...\n")

    for tracker_cfg, conf_thresh, iou_thresh in combos:
        print(
            f"Testing tracker={tracker_cfg}, conf={conf_thresh:.2f}, iou={iou_thresh:.2f}"
        )

        score, details = analyze_run(
            video_path=VIDEO_PATH,
            model_path=MODEL_PATH,
            tracker_cfg=tracker_cfg,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
        )

        all_rows.append(score)

        detail_name = (
            f"{safe_name(tracker_cfg)}__conf_{conf_thresh:.2f}__iou_{iou_thresh:.2f}.csv"
            .replace(".", "p")
        )
        write_details(os.path.join(DETAILS_DIR, detail_name), details)

    # Sort: fewer probable re-IDs, fewer short tracks, fewer unique IDs, longer average track
    all_rows.sort(
        key=lambda r: (
            r["probable_reid_events"],
            r["short_tracks"],
            r["total_unique_track_ids"],
            -r["avg_track_length"],
        )
    )

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tracker",
                "conf",
                "iou",
                "frames_processed",
                "total_vehicle_boxes",
                "total_unique_track_ids",
                "short_tracks",
                "avg_track_length",
                "min_track_length",
                "max_track_length",
                "fragmented_tracks",
                "total_internal_gaps",
                "probable_reid_events",
                "return_events",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nTop results:")
    for row in all_rows[:10]:
        print(
            f"tracker={row['tracker']}, conf={row['conf']:.2f}, iou={row['iou']:.2f}, "
            f"prob_reid={row['probable_reid_events']}, short={row['short_tracks']}, "
            f"unique_ids={row['total_unique_track_ids']}, avg_len={row['avg_track_length']}"
        )

    print(f"\nSaved summary to: {SUMMARY_CSV}")
    print(f"Saved per-run details to: {DETAILS_DIR}")


if __name__ == "__main__":
    main()