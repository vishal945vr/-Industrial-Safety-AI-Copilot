from langchain_core.prompts import ChatPromptTemplate
from .hybrid_sercher import retrieve_hybrid_info
from .model import llm as get_llm

llm = get_llm()

llm = get_llm()
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)





# Load LLM
llm = get_llm()


# ============================================================
# HELPER
# ============================================================

def to_query_string(data):

    if isinstance(data, str):
        return data

    if isinstance(data, dict):
        return "\n".join(
            f"{key}: {value}"
            for key, value in data.items()
        )

    if isinstance(data, list):
        return "\n".join(
            str(item)
            for item in data
        )

    return str(data)


# ============================================================
# SAFETY REPORT PROMPT
# ============================================================

report_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an Industrial Safety Compliance AI.

You analyze workplace safety incidents detected by
a Computer Vision system such as YOLO.

IMPORTANT RULES:

1. YOLO detection is evidence.
2. Never modify YOLO detection results.
3. Never invent PPE requirements.
4. Never invent company policies.
5. Never invent legal requirements.
6. Use the supplied risk_score and risk_level.
7. Do not create another risk score.
8. Use retrieved company safety policies as policy evidence.
9. If policy evidence is insufficient, say:

Policy breach cannot be confirmed from the available
policy evidence.

10. Give practical corrective actions.
11. Keep the report clear and professional.

Return a human-readable Industrial Safety Incident Report.
"""
    ),
    (
        "human",
        """
CURRENT INCIDENT:

{question}


RETRIEVED SAFETY POLICY:

{context}


Analyze the incident using ONLY the incident information
and retrieved policy evidence.
"""
    )
])


# ============================================================
# RETRIEVE POLICY CONTEXT
# ============================================================

def retrieve_policy_context(incident):

    if not incident:
        return []

    incident_text = to_query_string(incident)

    return retrieve_hybrid_info(incident_text)


# ============================================================
# GENERATE AI SAFETY REPORT
# ============================================================

def generate_safety_report(incident):

    if not incident:
        return "No safety incident is available."

    incident_text = to_query_string(incident)

    # RAG retrieval
    context = retrieve_hybrid_info(incident_text)

    # Convert documents into readable text
    if isinstance(context, list):

        context_text = "\n\n".join(
            getattr(doc, "page_content", str(doc))
            for doc in context
        )

    else:
        context_text = str(context)

    # Create prompt
    messages = report_prompt.invoke({
        "question": incident_text,
        "context": context_text
    })

    # Groq LLM
    response = llm.invoke(messages)

    return response.content


# ============================================================
# SAFETY COPILOT
# ============================================================

def ask_safety_copilot(question, incident):

    if not question:
        return "Please enter a question."

    if not incident:
        return "No current safety incident is available."

    incident_text = to_query_string(incident)

    # Combine user question + incident
    rag_query = f"""
User Question:
{question}

Current YOLO Safety Incident:
{incident_text}
"""

    # Retrieve company policy
    context = retrieve_hybrid_info(rag_query)

    if isinstance(context, list):

        context_text = "\n\n".join(
            getattr(doc, "page_content", str(doc))
            for doc in context
        )

    else:
        context_text = str(context)

    # Copilot prompt
    copilot_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an Industrial Safety Copilot.

Answer the user's question using:

1. Current YOLO incident.
2. Retrieved company safety policy.

Do NOT invent:

- company policies
- PPE requirements
- legal requirements
- emergency procedures
- detection results

If the retrieved policy does not provide enough evidence,
clearly say so.

Give a practical and safety-focused answer.
"""
        ),
        (
            "human",
            """
USER QUESTION:

{question}


CURRENT INCIDENT:

{incident}


RETRIEVED POLICY:

{context}
"""
        )
    ])

    messages = copilot_prompt.invoke({
        "question": question,
        "incident": incident_text,
        "context": context_text
    })

    # Call Groq
    response = llm.invoke(messages)

    return response.content

