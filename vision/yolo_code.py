"""
Industrial PPE Safety Detection System
----------------------------------------
Supports 3 input modes chosen by the user at runtime:
    1. Image file
    2. Video file
    3. Live camera (webcam)

For each detection, it computes a PPE risk score, risk level,
and safety action, then:
    - Overlays this info on the frame (video/camera modes)
    - Saves an incident JSON report
    - Auto-saves an evidence snapshot when risk is HIGH/CRITICAL
"""

import os
import json
import cv2
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# CONFIG
# ==========================================

MODEL_PATH = r"D:\rag_project\runs\detect\train-4\weights\best.pt"
OUTPUT_DIR = r"D:\rag_project\runs\detect\predict"
JSON_PATH = os.path.join(OUTPUT_DIR, "incident.json")
SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "evidence_snapshots")

CONF_THRESHOLD = 0.25
IMG_SIZE = 640

PPE_RISK = {
    "helmet":  {"weight": 40, "priority": 1, "risk": "Head injury"},
    "vest":    {"weight": 30, "priority": 2, "risk": "Low visibility / vehicle collision"},
    "gloves":  {"weight": 15, "priority": 3, "risk": "Hand injury"},
    "goggles": {"weight": 5,  "priority": 4, "risk": "Eye injury"},
    "boots":   {"weight": 10, "priority": 5, "risk": "Foot injury"},
    "mask":    {"weight": 5,  "priority": 6, "risk": "Dust / respiratory exposure"},
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# ==========================================
# RISK HELPERS
# ==========================================

def calculate_risk_level(score):
    if score >= 71:
        return "CRITICAL"
    elif score >= 41:
        return "HIGH"
    elif score >= 21:
        return "MEDIUM"
    else:
        return "LOW"


def safety_action_for(risk_level):
    return {
        "CRITICAL": "Stop work immediately and provide required PPE before allowing work.",
        "HIGH": "Correct PPE violation immediately before continuing work.",
        "MEDIUM": "Correct PPE violation and monitor the worker.",
        "LOW": "Continue monitoring PPE compliance.",
    }[risk_level]


def build_incident(detected, confidences, camera_id="CAM-01", incident_id="INC-001"):
    """Turn raw detection counts into a full incident report dict."""

    missing_ppe = [ppe for ppe in PPE_RISK if detected.get(ppe, 0) == 0]

    risk_score = 0
    risk_items = []

    for ppe in missing_ppe:
        info = PPE_RISK[ppe]
        risk_score += info["weight"]
        risk_items.append({
            "equipment": ppe,
            "priority": info["priority"],
            "risk_points": info["weight"],
            "predicted_risk": info["risk"],
        })

    risk_score = min(risk_score, 100)
    risk_items.sort(key=lambda x: x["priority"])
    risk_level = calculate_risk_level(risk_score)

    violation = ("Missing " + ", ".join(missing_ppe)) if missing_ppe else "No PPE violation"
    confidence = max(confidences.values()) if confidences else 0.0

    return {
        "incident_id": incident_id,
        "camera_id": camera_id,
        "timestamp": datetime.now().isoformat(),
        "person_count": detected.get("person", 0),
        "ppe_detected": {ppe: detected.get(ppe, 0) for ppe in PPE_RISK},
        "missing_ppe": missing_ppe,
        "violation": violation,
        "confidence": round(confidence, 2),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "priority_safety_equipment": risk_items,
        "safety_action": safety_action_for(risk_level),
    }


def save_incident(incident, path=JSON_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(incident, f, indent=4)


# ==========================================
# DETECTION CORE (shared by image/video/camera)
# ==========================================

def run_detection(model, frame):
    """
    Run YOLO on a single frame (numpy array or image path).
    Returns: annotated_frame, detected(dict), confidences(dict)
    """
    results = model.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False,
    )

    detected = {"helmet": 0, "vest": 0, "gloves": 0,
                "goggles": 0, "boots": 0, "mask": 0, "person": 0}
    confidences = {}

    result = results[0]
    names = result.names

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = names[class_id].lower().strip()

        if class_name == "person":
            detected["person"] += 1
        elif class_name in PPE_RISK:
            detected[class_name] += 1
            confidences[class_name] = max(confidences.get(class_name, 0), confidence)

    annotated_frame = result.plot()  # YOLO's built-in bounding-box drawing
    return annotated_frame, detected, confidences


def overlay_risk_info(frame, incident):
    """Draw risk score/level/missing PPE text on top of the annotated frame."""

    level = incident["risk_level"]
    color_map = {
        "CRITICAL": (0, 0, 255),   # red
        "HIGH": (0, 128, 255),     # orange
        "MEDIUM": (0, 255, 255),   # yellow
        "LOW": (0, 200, 0),        # green
    } 
    color = color_map.get(level, (255, 255, 255))

    lines = [
        f"Risk Level: {level}  (Score: {incident['risk_score']})",
        f"Persons: {incident['person_count']}",
        f"Missing: {', '.join(incident['missing_ppe']) if incident['missing_ppe'] else 'None'}",
    ]

    # Semi-transparent header bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    y = 25
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
        y += 28

    return frame


def maybe_save_snapshot(frame, incident):
    """Auto-save an evidence image whenever risk is HIGH or CRITICAL."""
    if incident["risk_level"] in ("HIGH", "CRITICAL"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SNAPSHOT_DIR, f"{incident['risk_level']}_{ts}.jpg")
        cv2.imwrite(path, frame)


# ==========================================
# MODE 1: IMAGE
# ==========================================

def run_on_image(model, image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None

    annotated, detected, confidences = run_detection(model, image_path)
    incident = build_incident(detected, confidences)
    annotated = overlay_risk_info(annotated, incident)

    save_incident(incident)
    maybe_save_snapshot(annotated, incident)

    print("\n========== SAFETY INCIDENT ==========")
    print(json.dumps(incident, indent=4))
    print(f"\nJSON saved: {JSON_PATH}")

    cv2.imshow("PPE Detection - Image", annotated)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return incident


# ==========================================
# MODE 2 & 3: VIDEO FILE / LIVE CAMERA
# (shared loop — source=0 for webcam, or a file path)
# ==========================================

def run_on_stream(model, source, window_title, camera_id):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Could not open source: {source}")
        return None

    print(f"✅ {window_title} started. Press Q to exit.")
    last_incident = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or frame read failed.")
            break

        frame_count += 1

        # Run detection every frame (reduce to every Nth frame if too slow,
        # e.g. `if frame_count % 3 == 0:` for lighter CPUs)
        annotated, detected, confidences = run_detection(model, frame)
        incident = build_incident(
            detected, confidences,
            camera_id=camera_id,
            incident_id=f"INC-{frame_count:05d}",
        )
        annotated = overlay_risk_info(annotated, incident)

        maybe_save_snapshot(annotated, incident)
        last_incident = incident

        cv2.imshow(window_title, annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if last_incident:
        save_incident(last_incident)
        print("\n========== LAST SAFETY INCIDENT ==========")
        print(json.dumps(last_incident, indent=4))
        print(f"\nJSON saved: {JSON_PATH}")

    return last_incident


def run_on_video(model, video_path):
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None
    return run_on_stream(model, video_path, "PPE Detection - Video", "CAM-VIDEO")


def run_on_camera(model, camera_index=0):
    return run_on_stream(model, camera_index, "Industrial Safety Camera", "CAM-LIVE")


# ==========================================
# MAIN MENU
# ==========================================

def main():
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded.\n")

    print("Select input source:")
    print("  1. Image")
    print("  2. Video file")
    print("  3. Live camera")

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        path = input("Enter image path: ").strip().strip('"')
        run_on_image(model, path)

    elif choice == "2":
        path = input("Enter video path: ").strip().strip('"')
        run_on_video(model, path)

    elif choice == "3":
        run_on_camera(model, camera_index=0)

    else:
        print("❌ Invalid choice. Exiting.")


if __name__ == "__main__":
    main()