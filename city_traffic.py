"""
 
 SMART CITY TRAFFIC MANAGEMENT SYSTEM
 
 
"""
 
import os
import sys
import time
import pickle
import random
import zipfile
import shutil
from dataclasses import dataclass
from typing import List
from collections import deque, defaultdict
 
import cv2
import numpy as np
 
# ============================================================================
# 1. CONFIG - EDIT THIS SECTION
# ============================================================================
 
# Path to the zip file you downloaded
ZIP_PATH = r"C:\Users\M SRI LIKITH RAHUL\Downloads\archive (1).zip"
 
# Folder to extract the zip into 
EXTRACT_DIR = r"C:\Users\M SRI LIKITH RAHUL\Downloads\archive_extracted"
 
 
VIDEO_PATH_OVERRIDE = None  # e.g. r"C:\Users\...\traffic.mp4"
 
# Where the annotated output video will be saved
OUTPUT_PATH = r"C:\Users\M SRI LIKITH RAHUL\Downloads\traffic_output.mp4"
 
# YOLO model (auto-downloaded on first run, ~6MB)
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
TRACKER_CONFIG = "bytetrack.yaml"
 
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PEDESTRIAN_CLASSES = {0: "person"}
ALL_TARGET_CLASSES = list(VEHICLE_CLASSES.keys()) + list(PEDESTRIAN_CLASSES.keys())
 
NUM_LANES = 4
LANE_NAMES = ["Lane-A (Top-Left)", "Lane-B (Top-Right)",
              "Lane-C (Bottom-Left)", "Lane-D (Bottom-Right)"]
 
DECISION_INTERVAL_FRAMES = 30
MIN_GREEN_FRAMES = 60
MAX_GREEN_FRAMES = 300
YELLOW_FRAMES = 15
 
RL_ALPHA = 0.10
RL_GAMMA = 0.90
RL_EPSILON_START = 0.25
RL_EPSILON_MIN = 0.05
RL_EPSILON_DECAY = 0.995
Q_TABLE_SAVE_PATH = "q_table.pkl"
 
DENSITY_LOW_MAX = 3
DENSITY_MED_MAX = 8
FLOW_WINDOW_SECONDS = 60
 
SHOW_LIVE_WINDOW = True
WRITE_OUTPUT_VIDEO = True
PRINT_PROGRESS_EVERY = 60
 
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")
 
 
 
def resolve_video_path() -> str:
    """Returns a real, playable video path. If VIDEO_PATH_OVERRIDE is set,
    uses that. Otherwise extracts ZIP_PATH into EXTRACT_DIR and finds the
    first video file inside it."""
    if VIDEO_PATH_OVERRIDE:
        return VIDEO_PATH_OVERRIDE
 
    if not os.path.isfile(ZIP_PATH):
        print(f"[ERROR] Zip file not found at: {ZIP_PATH}")
        print("        Edit ZIP_PATH near the top of this script.")
        sys.exit(1)
 
    
    if os.path.isdir(EXTRACT_DIR):
        for root, _dirs, files in os.walk(EXTRACT_DIR):
            for f in files:
                if f.lower().endswith(VIDEO_EXTENSIONS):
                    return os.path.join(root, f)
 
    print(f"[INFO] Extracting {ZIP_PATH} -> {EXTRACT_DIR}")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
 
    for root, _dirs, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f.lower().endswith(VIDEO_EXTENSIONS):
                found_path = os.path.join(root, f)
                print(f"[INFO] Found video: {found_path}")
                return found_path
 
    print("[ERROR] No video file (.mp4/.avi/.mov/.mkv/.wmv/.flv/.webm) was "
          "found inside the zip. Here is what WAS extracted:")
    for root, _dirs, files in os.walk(EXTRACT_DIR):
        for f in files:
            print("   ", os.path.join(root, f))
    sys.exit(1)
 
 
 
@dataclass
class Detection:
    track_id: int
    cls_id: int
    label: str
    is_pedestrian: bool
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
 
    @property
    def centroid(self):
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
 
 
class TrafficDetector:
    def __init__(self, model_path=MODEL_PATH, conf_threshold=CONF_THRESHOLD,
                 iou_threshold=IOU_THRESHOLD, tracker=TRACKER_CONFIG):
        from ultralytics import YOLO
        print(f"[Detector] Loading YOLO model '{model_path}' ...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker = tracker
        self.target_classes = ALL_TARGET_CLASSES
        print("[Detector] Model loaded.")
 
    def detect_and_track(self, frame) -> List[Detection]:
        results = self.model.track(
            frame, persist=True, tracker=self.tracker,
            classes=self.target_classes, conf=self.conf_threshold,
            iou=self.iou_threshold, verbose=False,
        )
 
        detections: List[Detection] = []
        if not results:
            return detections
 
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return detections
 
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
 
        for i in range(len(ids)):
            cls_id = int(cls_ids[i])
            is_ped = cls_id in PEDESTRIAN_CLASSES
            label = VEHICLE_CLASSES.get(cls_id, PEDESTRIAN_CLASSES.get(cls_id, "object"))
            x1, y1, x2, y2 = xyxy[i]
            detections.append(Detection(
                track_id=int(ids[i]), cls_id=cls_id, label=label,
                is_pedestrian=is_ped, conf=float(confs[i]),
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
            ))
        return detections
 
 
 
def assign_lane(cx, cy, frame_w, frame_h) -> int:
    left = cx < frame_w / 2
    top = cy < frame_h / 2
    if top and left:
        return 0
    if top and not left:
        return 1
    if not top and left:
        return 2
    return 3
 
 
def congestion_level(count: int) -> str:
    if count <= DENSITY_LOW_MAX:
        return "LOW"
    if count <= DENSITY_MED_MAX:
        return "MEDIUM"
    return "HIGH"
 
 
class TrafficAnalytics:
    def __init__(self, num_lanes: int, fps: float):
        self.num_lanes = num_lanes
        self.fps = max(fps, 1e-3)
        self.track_lane = {}
        self.wait_frames = defaultdict(int)
        self.flow_window = defaultdict(deque)
        self.seen_ids_per_lane = defaultdict(set)
        self.all_ids = set()
        self.all_pedestrian_ids = set()
 
    def update(self, detections, frame_w, frame_h, frame_idx, active_lane):
        video_time = frame_idx / self.fps
        lane_counts = {i: 0 for i in range(self.num_lanes)}
        pedestrian_count = 0
 
        for det in detections:
            cx, cy = det.centroid
            lane = assign_lane(cx, cy, frame_w, frame_h)
 
            if det.is_pedestrian:
                pedestrian_count += 1
                self.all_pedestrian_ids.add(det.track_id)
                continue
 
            lane_counts[lane] += 1
            self.track_lane[det.track_id] = lane
            self.all_ids.add(det.track_id)
 
            if det.track_id not in self.seen_ids_per_lane[lane]:
                self.seen_ids_per_lane[lane].add(det.track_id)
                self.flow_window[lane].append((video_time, det.track_id))
 
            if lane != active_lane:
                self.wait_frames[det.track_id] += 1
 
        for lane in range(self.num_lanes):
            dq = self.flow_window[lane]
            while dq and (video_time - dq[0][0]) > FLOW_WINDOW_SECONDS:
                dq.popleft()
 
        return {"lane_counts": lane_counts, "pedestrian_count": pedestrian_count,
                "video_time": video_time}
 
    def flow_rate_per_lane(self, lane: int) -> float:
        n = len(self.flow_window[lane])
        minutes = FLOW_WINDOW_SECONDS / 60.0
        return n / minutes if minutes > 0 else 0.0
 
    def total_flow_rate(self) -> float:
        return sum(self.flow_rate_per_lane(i) for i in range(self.num_lanes))
 
    def avg_wait_time_seconds(self, lane: int, current_lane_ids) -> float:
        ids = [tid for tid in current_lane_ids if self.track_lane.get(tid) == lane]
        if not ids:
            return 0.0
        total = sum(self.wait_frames[tid] for tid in ids)
        return (total / len(ids)) / self.fps
 
    def total_unique_vehicles(self) -> int:
        return len(self.all_ids)
 
    def total_unique_pedestrians(self) -> int:
        return len(self.all_pedestrian_ids)
 
 
 
def _discretize(count: int) -> int:
    if count <= DENSITY_LOW_MAX:
        return 0
    if count <= DENSITY_MED_MAX:
        return 1
    return 2
 
 
class QLearningTrafficAgent:
    def __init__(self, num_lanes=NUM_LANES, alpha=RL_ALPHA, gamma=RL_GAMMA,
                 epsilon=RL_EPSILON_START, epsilon_min=RL_EPSILON_MIN,
                 epsilon_decay=RL_EPSILON_DECAY, q_table_path=Q_TABLE_SAVE_PATH):
        self.num_lanes = num_lanes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table_path = q_table_path
        self.q_table = defaultdict(lambda: np.zeros(self.num_lanes))
        self._load()
        self.prev_state = None
        self.prev_action = None
 
    def _load(self):
        if self.q_table_path and os.path.isfile(self.q_table_path):
            try:
                with open(self.q_table_path, "rb") as f:
                    loaded = pickle.load(f)
                for k, v in loaded.items():
                    self.q_table[k] = np.array(v)
                print(f"[RL Agent] Loaded existing Q-table ({len(loaded)} states)")
            except Exception as e:
                print(f"[RL Agent] Could not load Q-table ({e}); starting fresh.")
 
    def save(self):
        if not self.q_table_path:
            return
        try:
            with open(self.q_table_path, "wb") as f:
                pickle.dump(dict(self.q_table), f)
            print(f"[RL Agent] Saved Q-table ({len(self.q_table)} states)")
        except Exception as e:
            print(f"[RL Agent] Failed to save Q-table: {e}")
 
    def get_state(self, lane_counts: dict, active_lane: int):
        densities = tuple(_discretize(lane_counts.get(i, 0)) for i in range(self.num_lanes))
        return densities + (active_lane,)
 
    def choose_action(self, state) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.num_lanes)
        q_values = self.q_table[state]
        return int(np.argmax(q_values))
 
    def compute_reward(self, lane_counts: dict, active_lane: int) -> float:
        total_queue = sum(lane_counts.values())
        served_bonus = 2.0 * lane_counts.get(active_lane, 0)
        return served_bonus - total_queue
 
    def update(self, state, action, reward, next_state):
        q_values = self.q_table[state]
        next_q_values = self.q_table[next_state]
        best_next = np.max(next_q_values)
        q_values[action] += self.alpha * (reward + self.gamma * best_next - q_values[action])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
 
    def decide(self, lane_counts: dict, active_lane: int) -> int:
        state = self.get_state(lane_counts, active_lane)
        reward = self.compute_reward(lane_counts, active_lane)
        if self.prev_state is not None:
            self.update(self.prev_state, self.prev_action, reward, state)
        action = self.choose_action(state)
        self.prev_state = state
        self.prev_action = action
        return action
 

 
class SignalController:
    def __init__(self, num_lanes=NUM_LANES, min_green=MIN_GREEN_FRAMES,
                 max_green=MAX_GREEN_FRAMES, yellow_frames=YELLOW_FRAMES):
        self.num_lanes = num_lanes
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_frames = yellow_frames
        self.active_lane = 0
        self.phase = "GREEN"
        self.timer = 0
        self.next_lane = None
        self.desired_lane = 0
 
    def set_desired_lane(self, lane_idx: int):
        self.desired_lane = lane_idx
 
    def step(self):
        self.timer += 1
        if self.phase == "GREEN":
            can_switch = self.timer >= self.min_green
            must_switch = self.timer >= self.max_green
            wants_switch = self.desired_lane != self.active_lane
            if must_switch or (can_switch and wants_switch):
                self.next_lane = (self.desired_lane if self.desired_lane != self.active_lane
                                   else (self.active_lane + 1) % self.num_lanes)
                self.phase = "YELLOW"
                self.timer = 0
        elif self.phase == "YELLOW":
            if self.timer >= self.yellow_frames:
                self.active_lane = self.next_lane
                self.phase = "GREEN"
                self.timer = 0
        return self.active_lane, self.phase, self.timer
 
    def lane_state(self, lane_idx: int) -> str:
        if lane_idx != self.active_lane:
            return "RED"
        return "YELLOW" if self.phase == "YELLOW" else "GREEN"
 
 

 
COLOR_VEHICLE = (60, 200, 60)
COLOR_PEDESTRIAN = (60, 160, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL_BG = (20, 20, 20)
 
CONGESTION_COLOR = {"LOW": (60, 200, 60), "MEDIUM": (0, 200, 255), "HIGH": (0, 0, 255)}
SIGNAL_COLOR = {"GREEN": (0, 220, 0), "YELLOW": (0, 220, 220), "RED": (0, 0, 220)}
 
 
def draw_detections(frame, detections):
    for det in detections:
        color = COLOR_PEDESTRIAN if det.is_pedestrian else COLOR_VEHICLE
        p1 = (int(det.x1), int(det.y1))
        p2 = (int(det.x2), int(det.y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        label = f"{det.label} ID:{det.track_id} {det.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 8), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(frame, label, (p1[0] + 2, p1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame
 
 
def draw_lane_density_overlay(frame, lane_counts, num_lanes=4, alpha=0.18):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    quadrants = [(0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2),
                 (0, h // 2, w // 2, h), (w // 2, h // 2, w, h)]
    for lane_idx, (x1, y1, x2, y2) in enumerate(quadrants[:num_lanes]):
        count = lane_counts.get(lane_idx, 0)
        level = congestion_level(count)
        color = CONGESTION_COLOR[level]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.line(frame, (w // 2, 0), (w // 2, h), (90, 90, 90), 1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (90, 90, 90), 1)
    return frame
 
 
def draw_signal_lights(frame, signal_controller, num_lanes=4):
    h, w = frame.shape[:2]
    centers = [(w // 4, h // 4), (3 * w // 4, h // 4),
               (w // 4, 3 * h // 4), (3 * w // 4, 3 * h // 4)]
    for lane_idx in range(num_lanes):
        cx, cy = centers[lane_idx]
        state = signal_controller.lane_state(lane_idx)
        color = SIGNAL_COLOR[state]
        cv2.circle(frame, (cx, cy - 30), 14, (30, 30, 30), -1)
        cv2.circle(frame, (cx, cy - 30), 14, color, 3)
        cv2.circle(frame, (cx, cy - 30), 9, color, -1)
        cv2.putText(frame, LANE_NAMES[lane_idx].split(" ")[0], (cx - 30, cy - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    return frame
 
 
def _put_line(frame, text, x, y, scale=0.55, color=COLOR_TEXT, thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
 
 
def draw_dashboard_panel(frame, stats: dict):
    h, w = frame.shape[:2]
    panel_w = 340
    panel_h = min(40 + 26 * NUM_LANES + 150, h - 20)
    x0, y0 = 10, 10
 
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (90, 90, 90), 1)
 
    tx, ty = x0 + 12, y0 + 24
    _put_line(frame, "SMART TRAFFIC AI - LIVE DASHBOARD", tx, ty, 0.55, (255, 255, 255), 1)
    ty += 24
    _put_line(frame, f"Total vehicles tracked: {stats['total_vehicles']}", tx, ty)
    ty += 20
    _put_line(frame, f"Total pedestrians tracked: {stats['total_pedestrians']}"
              f"  (now: {stats['current_pedestrians']})", tx, ty)
    ty += 20
    _put_line(frame, f"Overall flow rate: {stats['total_flow']:.1f} veh/min", tx, ty)
    ty += 20
    active_name = LANE_NAMES[stats['active_lane']]
    _put_line(frame, f"Signal: {active_name} -> {stats['phase']} "
              f"({stats['phase_timer_sec']:.1f}s)", tx, ty, 0.5, (0, 220, 220))
    ty += 26
 
    _put_line(frame, "Per-lane status:", tx, ty, 0.5, (200, 200, 200))
    ty += 20
    for lane_idx in range(NUM_LANES):
        count = stats["lane_counts"].get(lane_idx, 0)
        level = stats["lane_congestion"].get(lane_idx, "LOW")
        wait = stats["lane_wait"].get(lane_idx, 0.0)
        flow = stats["lane_flow"].get(lane_idx, 0.0)
        color = CONGESTION_COLOR[level]
        name = LANE_NAMES[lane_idx].split(" ")[0]
        line = f"{name}: {count} veh | {level:6s} | wait {wait:4.1f}s | {flow:4.1f}/min"
        _put_line(frame, line, tx, ty, 0.48, color)
        ty += 22
    return frame
 
 
def render_frame(frame, detections, analytics_frame_stats, analytics, signal_controller, fps):
    lane_counts = analytics_frame_stats["lane_counts"]
 
    frame = draw_lane_density_overlay(frame, lane_counts, NUM_LANES)
    frame = draw_detections(frame, detections)
    frame = draw_signal_lights(frame, signal_controller, NUM_LANES)
 
    lane_congestion, lane_wait, lane_flow = {}, {}, {}
    current_ids = [d.track_id for d in detections if not d.is_pedestrian]
    for lane_idx in range(NUM_LANES):
        count = lane_counts.get(lane_idx, 0)
        lane_congestion[lane_idx] = congestion_level(count)
        lane_wait[lane_idx] = analytics.avg_wait_time_seconds(lane_idx, current_ids)
        lane_flow[lane_idx] = analytics.flow_rate_per_lane(lane_idx)
 
    stats = {
        "total_vehicles": analytics.total_unique_vehicles(),
        "total_pedestrians": analytics.total_unique_pedestrians(),
        "current_pedestrians": analytics_frame_stats["pedestrian_count"],
        "active_lane": signal_controller.active_lane,
        "phase": signal_controller.phase,
        "phase_timer_sec": signal_controller.timer / fps,
        "lane_counts": lane_counts,
        "lane_congestion": lane_congestion,
        "lane_wait": lane_wait,
        "lane_flow": lane_flow,
        "total_flow": analytics.total_flow_rate(),
        "fps_actual": fps,
    }
    frame = draw_dashboard_panel(frame, stats)
    return frame
 
 
def main():
    video_path = resolve_video_path()
    print(f"[Main] Using video: {video_path}")
 
    if not os.path.isfile(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)
 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] OpenCV could not open the video: {video_path}")
        sys.exit(1)
 
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Main] Video opened: {width}x{height} @ {fps:.2f}fps, ~{total_frames} frames")
 
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
 
    writer = None
    if WRITE_OUTPUT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"[WARNING] Could not open VideoWriter for {OUTPUT_PATH}. "
                  "Output video will not be saved.")
            writer = None
 
    detector = TrafficDetector()
    analytics = TrafficAnalytics(num_lanes=NUM_LANES, fps=fps)
    rl_agent = QLearningTrafficAgent()
    signal = SignalController()
 
    frame_idx = 0
    start_time = time.time()
 
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
 
            detections = detector.detect_and_track(frame)
            frame_stats = analytics.update(detections, width, height, frame_idx, signal.active_lane)
            lane_counts = frame_stats["lane_counts"]
 
            if frame_idx % DECISION_INTERVAL_FRAMES == 0:
                recommended_lane = rl_agent.decide(lane_counts, signal.active_lane)
                signal.set_desired_lane(recommended_lane)
 
            signal.step()
 
            out_frame = render_frame(frame, detections, frame_stats, analytics, signal, fps)
 
            if writer is not None:
                writer.write(out_frame)
 
            if SHOW_LIVE_WINDOW:
                cv2.imshow("Smart Traffic AI", out_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[Main] 'q' pressed, stopping early.")
                    break
 
            frame_idx += 1
            if frame_idx % PRINT_PROGRESS_EVERY == 0:
                elapsed = time.time() - start_time
                proc_fps = frame_idx / elapsed if elapsed > 0 else 0
                pct = (frame_idx / total_frames * 100) if total_frames else 0
                print(f"[Main] Frame {frame_idx}/{total_frames} ({pct:5.1f}%) | "
                      f"processing {proc_fps:.1f} fps | "
                      f"vehicles so far: {analytics.total_unique_vehicles()}")
 
    except KeyboardInterrupt:
        print("[Main] Interrupted by user.")
 
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if SHOW_LIVE_WINDOW:
            cv2.destroyAllWindows()
        rl_agent.save()
 
        elapsed = time.time() - start_time
        print("\n[Main] Done.")
        print(f"       Frames processed : {frame_idx}")
        print(f"       Time elapsed     : {elapsed:.1f}s")
        print(f"       Unique vehicles  : {analytics.total_unique_vehicles()}")
        print(f"       Unique pedestrians: {analytics.total_unique_pedestrians()}")
        if writer is not None:
            print(f"       Output video saved to: {OUTPUT_PATH}")
 
 
if __name__ == "__main__":
    main()
 