
from ultralytics import YOLO
import json
from datetime import datetime
# ----------------------------------
imagepath = (
        r"D:\rag_project\datasets\construction-ppe"
        r"\images\test\viedo2.mp4"
       
    )


# ==========================================
# PPE RISK CONFIGURATION


PPE_RISK = {
    "helmet": {
        "weight": 40,
        "priority": 1,
        "risk": "Head injury"
    },

    "vest": {
        "weight": 30,
        "priority": 2,
        "risk": "Low visibility / vehicle collision"
    },

    "gloves": {
        "weight": 15,
        "priority": 3,
        "risk": "Hand injury"
    },

    "goggles": {
        "weight": 5,
        "priority": 4,
        "risk": "Eye injury"
    },

    "boots": {
        "weight": 10,
        "priority": 5,
        "risk": "Foot injury"
    },

    "mask": {
        "weight": 5,
        "priority": 6,
        "risk": "Dust / respiratory exposure"
    }
}


# risk level 

def calculate_risk_level(score):

    if score >= 71:
        return "CRITICAL"

    elif score >= 41:
        return "HIGH"

    elif score >= 21:
        return "MEDIUM"

    else:
        return "LOW"


# ==========================================
# PREDICT PPE

def predict(imagescr):

# model load 
    model = YOLO(
        r"D:\rag_project\runs\detect\train-4\weights\best.pt"
    )
# image load 
    image_path = imagescr
    # YOLO prediction
    results = model.predict(
        source=image_path,
        conf=0.25,
        imgsz=640,
        save=True,
        show=True
    )



    detected = {
        "helmet": 0,
        "vest": 0,
        "gloves": 0,
        "goggles": 0,
        "boots": 0,
        "mask": 0,
        "person": 0
    }

# values

    confidences = {}

    # Process YOLO results
  

    for result in results:

        names = result.names

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = names[class_id].lower().strip()

            # Person
            if class_name == "person":

                detected["person"] += 1

            # PPE
            elif class_name in PPE_RISK:

                detected[class_name] += 1

                # Keep highest confidence
                if class_name not in confidences:

                    confidences[class_name] = confidence

                else:

                    confidences[class_name] = max(
                        confidences[class_name],
                        confidence
                    )


    missing_ppe = []

    for ppe in PPE_RISK:

        if detected[ppe] == 0:

            missing_ppe.append(ppe)

    # CALCULATE RISK SCORE
   
    risk_score = 0

    risk_items = []

    for ppe in missing_ppe:

        weight = PPE_RISK[ppe]["weight"]

        priority = PPE_RISK[ppe]["priority"]

        risk_description = PPE_RISK[ppe]["risk"]

        risk_score += weight

        risk_items.append({
            "equipment": ppe,
            "priority": priority,
            "risk_points": weight,
            "predicted_risk": risk_description
        })

    # Maximum score = 100
    risk_score = min(risk_score, 100)

    # SORT BY PRIORITY

    risk_items.sort(
        key=lambda x: x["priority"]
    )

    # RISK LEVEL
   

    risk_level = calculate_risk_level(
        risk_score
    )

   
    # OVERALL VIOLATION
  

    if missing_ppe:

        violation = "Missing " + ", ".join(
            missing_ppe
        )

    else:

        violation = "No PPE violation"

    # ======================================
    # OVERALL CONFIDENCE
    # ======================================

    if confidences:

        confidence = max(
            confidences.values()
        )

    else:

        confidence = 0.0

    # ======================================
    # SAFETY ACTION
    # ======================================

    if risk_level == "CRITICAL":

        safety_action = (
            "Stop work immediately and provide "
            "required PPE before allowing work."
        )

    elif risk_level == "HIGH":

        safety_action = (
            "Correct PPE violation immediately "
            "before continuing work."
        )

    elif risk_level == "MEDIUM":

        safety_action = (
            "Correct PPE violation and monitor "
            "the worker."
        )

    else:

        safety_action = (
            "Continue monitoring PPE compliance."
        )

  
    # INCIDENT JSON


    incident = {

        "incident_id": "INC-001",

        "camera_id": "CAM-01",

        "timestamp": datetime.now().isoformat(),

        "person_count": detected["person"],

        "ppe_detected": {

            "helmet": detected["helmet"],
            "vest": detected["vest"],
            "gloves": detected["gloves"],
            "goggles": detected["goggles"],
            "boots": detected["boots"],
            "mask": detected["mask"]
        },

        "missing_ppe": missing_ppe,

        "violation": violation,

        "confidence": round(
            confidence,
            2
        ),

        "risk_score": risk_score,

        "risk_level": risk_level,

        "priority_safety_equipment": risk_items,

        "safety_action": safety_action
    }

   
    # SAVE JSON
   

    json_path = (
        r"D:\rag_project\runs\detect\predict"
        r"\incident.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            incident,
            file,
            indent=4
        )

    # ======================================
    # PRINT RESULT
    # ======================================

    print("\n========== SAFETY INCIDENT ==========")

    print(
        json.dumps(
            incident,
            indent=4
        )
    )

    print(
        f"\nJSON saved: {json_path}"
    )

    return incident



# MAIN


if __name__ == "__main__":

    predict(imagepath)

