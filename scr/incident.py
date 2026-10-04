from model import get_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_core.runnables import RunnableLambda
import json
# llm 
prompt_llm = get_llm()

# prompt 
incident_prompt_template = PromptTemplate(
    input_variables=[
        "incident_id", "camera_id", "timestamp", "person_count",
        "ppe_detected_str", "missing_ppe_str", "violation",
        "confidence", "risk_score", "risk_level",
        "priority_str", "safety_action"
    ],
    template="""You are a workplace safety compliance analyst reviewing an automated
PPE (Personal Protective Equipment) detection report from a construction/industrial site.

INCIDENT DETAILS
-----------------
Incident ID   : {incident_id}
Camera ID     : {camera_id}
Timestamp     : {timestamp}
Workers Detected : {person_count}

PPE DETECTION SUMMARY (count per equipment type)
-----------------
{ppe_detected_str}

VIOLATION DETECTED
-----------------
Missing PPE       : {missing_ppe_str}
Violation Summary : {violation}
Detection Confidence : {confidence}

RISK ASSESSMENT
-----------------
Risk Score : {risk_score} / 100
Risk Level : {risk_level}

PRIORITY-RANKED MISSING EQUIPMENT (highest priority first)
-----------------
{priority_str}

RECOMMENDED IMMEDIATE ACTION
-----------------
{safety_action}

TASK
-----------------
Based on the above incident data, respond with a JSON object containing exactly these keys:
- "summary": A concise, human-readable safety incident summary (2-3 sentences) suitable for a supervisor.
- "escalation_level": One of "Low", "Medium", "High", "Critical", with a 1-line justification.
- "corrective_actions": A prioritized list of corrective actions addressing the missing PPE items in order of risk.
- "compliance_note": A short compliance note suitable for logging in a safety audit report.

Respond with ONLY the JSON object, no extra text.
"""
)
# output
json_output = StrOutputParser()
# function 
def preprocess_incident(data: dict) -> dict:
    ppe_detected_str = "\n".join(
        f"  - {item.capitalize()}: {count}" for item, count in data["ppe_detected"].items()
    )
    missing_ppe_str = ", ".join(item.capitalize() for item in data["missing_ppe"])
    priority_str = "\n".join(
        f"  {i+1}. {p['equipment'].capitalize()} "
        f"(Priority: {p['priority']}, Risk Points: {p['risk_points']}, "
        f"Predicted Risk: {p['predicted_risk']})"
        for i, p in enumerate(data["priority_safety_equipment"])
    )
    return {
        "incident_id": data["incident_id"],
        "camera_id": data["camera_id"],
        "timestamp": data["timestamp"],
        "person_count": data["person_count"],
        "ppe_detected_str": ppe_detected_str,
        "missing_ppe_str": missing_ppe_str,
        "violation": data["violation"],
        "confidence": f"{data['confidence'] * 100:.1f}%",
        "risk_score": data["risk_score"],
        "risk_level": data["risk_level"],
        "priority_str": priority_str,
        "safety_action": data["safety_action"],
    }


# chain
incident_model = RunnableLambda(preprocess_incident) | incident_prompt_template | prompt_llm | json_output

# json 
json_path = r"D:\rag_project\runs\detect\predict\incident.json"
try:
    with open(json_path, "r") as file:
        incident_json = json.load(file)
except FileNotFoundError:
    print(f" file does not exist: {json_path}")
    incident_json = None

result_yolo_incident = incident_model.invoke(incident_json)

# function rag ko prompt de ga 

def rag_propmt():
    result_yolo_incident = incident_model.invoke(incident_json)
    return result_yolo_incident
    

    

