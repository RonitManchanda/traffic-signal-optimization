import os
import csv
import math
from copy import deepcopy
from collections import defaultdict, deque
from itertools import product

import cv2
import yaml
from ultralytics import YOLO


# =========================
# USER CONFIG
# =========================
VIDEO_PATH = "IntersectionCropped.mp4"
MODEL_PATH = "yolo11s.pt"

# Base tracker config to mutate
BASE_TRACKER_YAML = "botsort_custom.yaml"

# Detector settings (keep fixed while tuning tracker yaml)
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45

# Vehicle classes only
TARGET_CLASSES = {2, 3, 5, 7}

# Ignore tiny detections
MIN_BOX_AREA = 900

# Re-ID heuristics for evaluation
MAX_GAP_FOR_RETURN = 20
RETURN_DISTANCE_THRESHOLD = 90
CENTER_MATCH_DISTANCE = 70
SHORT_TRACK_MAX_LEN = 5

# Output
SUMMARY_CSV = "botsort_yaml_tuning_summary.csv"
DETAILS_DIR = "botsort_yaml_tuning_details"
TEMP_TRACKER_DIR = "temp_botsort_configs"

# =========================
# PARAMETER GRID
# Start small first
# =========================
PARAM_GRID = {
    "track_high_thresh": [0.15, 0.18],
    "track_low_thresh": [0.08],
    "new_track_thresh": [0.30, 0.35],
    "track_buffer": [40, 50, 60],
    "match_thresh": [0.80, 0.85, 0.90],
}

# Fixed params from your custom yaml
FIXED_OVERRIDES = {
    "tracker_type": "botsort",
    "fuse_score": True,
    "gmc_method": "sparseOptFlow",
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.8,
    "with_reid": False,
    "model": "auto",
}


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
        .replace(":", "_")
    )


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def combo_name(params):
    return (
        f"thigh_{params['track_high_thresh']}"
        f"__tlow_{params['track_low_thresh']}"
        f"__new_{params['new_track_thresh']}"
        f"__buf_{params['track_buffer']}"
        f"__match_{params['match_thresh']}"
    ).replace(".", "p")


# =========================
# CORE ANALYSIS
# =========================
def analyze_run(video_path, model_path, tracker_cfg, conf_thresh, iou_thresh):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_index = 0

    track_frames = defaultdict(list)
    track_centers = defaultdict(list)
    track_classes = {}
    last_seen = {}
    last_center = {}

    total_vehicle_boxes = 0
    probable_reid_events = 0
    return_events = 0
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

        active_ids = {tid for tid, _, _ in current_tracks}

        for tid, seen_frame in list(last_seen.items()):
            if tid in active_ids:
                continue

            if seen_frame == frame_index - 1:
                recently_lost.append(
                    (tid, track_classes.get(tid), seen_frame, last_center.get(tid))
                )

        while recently_lost and frame_index - recently_lost[0][2] > MAX_GAP_FOR_RETURN:
            recently_lost.popleft()

        for i in range(len(current_tracks)):
            id1, cls1, c1 = current_tracks[i]
            for j in range(i + 1, len(current_tracks)):
                id2, cls2, c2 = current_tracks[j]
                if cls1 != cls2:
                    continue
                if euclidean(c1, c2) < CENTER_MATCH_DISTANCE:
                    pass

        frame_index += 1

    cap.release()

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
        "tracker_cfg": tracker_cfg,
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


# =========================
# MAIN
# =========================
def main():
    os.makedirs(DETAILS_DIR, exist_ok=True)
    os.makedirs(TEMP_TRACKER_DIR, exist_ok=True)

    base_cfg = load_yaml(BASE_TRACKER_YAML)

    fieldnames = [
        "tracker_cfg",
        "track_high_thresh",
        "track_low_thresh",
        "new_track_thresh",
        "track_buffer",
        "match_thresh",
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
    ]

    all_rows = []

    combos = list(product(
        PARAM_GRID["track_high_thresh"],
        PARAM_GRID["track_low_thresh"],
        PARAM_GRID["new_track_thresh"],
        PARAM_GRID["track_buffer"],
        PARAM_GRID["match_thresh"],
    ))

    print(f"Running {len(combos)} BoT-SORT YAML tuning tests...\n")

    summary_file = open(SUMMARY_CSV, "w", newline="")
    writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
    writer.writeheader()
    summary_file.flush()

    for thigh, tlow, newt, tbuf, mthresh in combos:
        params = {
            "track_high_thresh": thigh,
            "track_low_thresh": tlow,
            "new_track_thresh": newt,
            "track_buffer": tbuf,
            "match_thresh": mthresh,
        }

        cfg = deepcopy(base_cfg)
        cfg.update(FIXED_OVERRIDES)
        cfg.update(params)

        cfg_name = combo_name(params)
        cfg_path = os.path.join(TEMP_TRACKER_DIR, f"{cfg_name}.yaml")
        save_yaml(cfg_path, cfg)

        print(
            f"Testing {cfg_name} | "
            f"high={thigh:.2f}, low={tlow:.2f}, new={newt:.2f}, "
            f"buffer={tbuf}, match={mthresh:.2f}"
        )

        score, details = analyze_run(
            video_path=VIDEO_PATH,
            model_path=MODEL_PATH,
            tracker_cfg=cfg_path,
            conf_thresh=CONF_THRESHOLD,
            iou_thresh=IOU_THRESHOLD,
        )

        row = {
            "tracker_cfg": cfg_name,
            "track_high_thresh": thigh,
            "track_low_thresh": tlow,
            "new_track_thresh": newt,
            "track_buffer": tbuf,
            "match_thresh": mthresh,
            **score,
        }

        all_rows.append(row)
        writer.writerow(row)
        summary_file.flush()

        details_path = os.path.join(DETAILS_DIR, f"{cfg_name}.csv")
        write_details(details_path, details)

        print(
            f"  -> reid={row['probable_reid_events']}, "
            f"short={row['short_tracks']}, "
            f"ids={row['total_unique_track_ids']}, "
            f"avg_len={row['avg_track_length']}"
        )

    summary_file.close()

    all_rows.sort(
        key=lambda r: (
            r["probable_reid_events"],
            r["short_tracks"],
            r["total_unique_track_ids"],
            -r["avg_track_length"],
        )
    )

    print("\nTop results:")
    for row in all_rows[:10]:
        print(
            f"{row['tracker_cfg']} | "
            f"reid={row['probable_reid_events']} | "
            f"short={row['short_tracks']} | "
            f"ids={row['total_unique_track_ids']} | "
            f"avg={row['avg_track_length']}"
        )

    print(f"\nSaved summary to: {SUMMARY_CSV}")
    print(f"Saved detail CSVs to: {DETAILS_DIR}")
    print(f"Saved temp tracker YAMLs to: {TEMP_TRACKER_DIR}")


if __name__ == "__main__":
    main()