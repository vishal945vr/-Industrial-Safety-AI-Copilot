<<<<<<< HEAD

# 🦺 Industrial Safety Copilot

An AI-powered **Industrial Safety Monitoring and Compliance Assistant** that combines **Computer Vision, YOLO, RAG, and LLMs** to detect workplace safety violations, analyze PPE compliance, retrieve relevant safety policies, and generate actionable safety reports.

---

## 📌 Project Overview

Industrial Safety Copilot is designed to assist safety teams in monitoring workplace environments.

The system uses:

* **YOLO** for real-time PPE and person detection
* **Computer Vision** for visual safety inspection
* **RAG (Retrieval-Augmented Generation)** for retrieving relevant safety policies
* **LLM** for safety analysis, AI reports, and conversational assistance
* **Flask** for the web application backend
* **Custom Web Dashboard** for monitoring incidents and visual evidence

The application can analyze uploaded images and monitor a live camera feed.

---

# 🚀 Key Features

## 1. 🖼️ Image Safety Analysis

Upload a workplace image and the system analyzes it using YOLO.

It can detect PPE such as:

* Helmet
* Safety Vest
* Gloves
* Goggles
* Boots
* Mask
* Person

The system generates an incident containing:

* Person count
* Detected PPE
* Missing PPE
* PPE violation
* Detection confidence
* Risk score
* Risk level
* Priority safety equipment
* Recommended safety action

---

## 2. 🎥 Live Camera Monitoring

The application supports live camera monitoring.

The camera feed is processed using YOLO to identify safety violations in real time.

The dashboard can display:

* Live video
* Current incident
* Risk level
* PPE status
* Person count
* Detection information

When the camera is stopped, the latest visual evidence snapshot can be displayed in the **Visual Evidence** section.

---

## 3. 🔍 Visual Evidence

Every important safety detection can be associated with an annotated image.

The evidence can contain:

* Detected persons
* PPE detections
* Bounding boxes
* Risk information
* Safety violation information

This provides visual support for the generated safety incident.

---

## 4. 🤖 Safety Copilot

The Safety Copilot allows users to ask questions about the current safety incident.

Example questions:

```text
Why is this incident considered high risk?
```

```text
What PPE is missing?
```

```text
What corrective action should be taken?
```

The Copilot uses:

1. Current YOLO incident
2. Retrieved safety policy
3. LLM reasoning

to generate the response.

---

## 5. 📄 AI Safety Report

The system can generate an AI-powered Industrial Safety Incident Report.

The report uses:

* YOLO detection results
* Incident information
* Risk level
* Retrieved company safety policies

The system is designed not to invent safety policies or detection results.

---

# 🧠 AI Architecture

```text
                    ┌─────────────────────┐
                    │       User          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Flask Web App     │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │ Image / Camera │        │ Safety Copilot  │
        └────────┬────────┘        └────────┬────────┘
                 │                          │
                 ▼                          ▼
        ┌─────────────────┐        ┌─────────────────┐
        │      YOLO       │        │   Hybrid RAG    │
        │ Computer Vision │        │ Policy Search   │
        └────────┬────────┘        └────────┬────────┘
                 │                          │
                 ▼                          ▼
        ┌─────────────────┐        ┌─────────────────┐
        │    Incident     │        │ Policy Context  │
        │    Analysis     │        └────────┬────────┘
        └────────┬────────┘                 │
                 │                          │
                 └─────────────┬────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │        LLM          │
                    │ Safety Reasoning    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Dashboard / Report  │
                    └─────────────────────┘
```

---

# 🏗️ Project Structure

```text
RAG_Project/
│
├── frontend/
│   └── app.py
│
├── scr/
│   ├── __init__.py
│   ├── model.py
│   ├── retrive.py
│   ├── hybrid_sercher.py
│   └── vectordatabase.py
│
├── vision/
│   └── yolo_code.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── runs/
│   └── detect/
│       └── train-4/
│           └── weights/
│               └── best.pt
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .env
└── README.md
```

---

# 🔧 Technologies Used

| Technology          | Purpose                      |
| ------------------- | ---------------------------- |
| Python              | Core programming language    |
| Flask               | Web application backend      |
| YOLO                | Object and PPE detection     |
| OpenCV              | Image and camera processing  |
| LangChain           | LLM and RAG orchestration    |
| Groq                | LLM inference                |
| HuggingFace         | Embeddings                   |
| FAISS               | Vector similarity search     |
| HTML/CSS/JavaScript | Frontend dashboard           |
| Docker              | Application containerization |

---

# 🧠 YOLO Safety Detection

The YOLO model is trained to detect workplace safety objects.

Example PPE classes:

```text
person
helmet
vest
gloves
goggles
boots
mask
```

The application uses the detection results to identify potential PPE violations.

---

# ⚠️ Risk Assessment

The system calculates a risk score based on missing PPE.

Example PPE weights:

```text
Helmet   → 40
Vest     → 30
Gloves   → 15
Boots    → 10
Goggles  → 5
Mask     → 5
```

Risk levels:

```text
71+       → CRITICAL
41 - 70   → HIGH
21 - 40   → MEDIUM
0 - 20    → LOW
```

The risk score is generated from the application's configured PPE rules and is not independently generated by the LLM.

---

# 📚 RAG Pipeline

The RAG system retrieves relevant safety policy information before generating an AI response.

```text
User Question
      │
      ▼
Hybrid Search
      │
      ├── Vector Search
      │
      └── Keyword / Retrieval Search
      │
      ▼
Relevant Safety Documents
      │
      ▼
Context
      │
      ▼
LLM
      │
      ▼
Safety Response
```

This helps the LLM ground its response in the available safety-policy documents.

---

# 🤖 LLM Pipeline

The LLM is used for:

* Safety Copilot
* Incident explanation
* AI safety reports
* Corrective action suggestions
* Policy-based analysis

The application passes the detected incident and retrieved policy context to the LLM.

The LLM should not modify the original YOLO detection results.

---

# 🌐 Flask API

Important application endpoints include:

```text
GET  /
```

Loads the main dashboard.

```text
POST /api/analyze
```

Analyzes an uploaded image.

```text
POST /api/camera/start
```

Starts camera processing.

```text
POST /api/camera/stop
```

Stops camera processing and can return the latest visual evidence.

```text
GET /video_feed
```

Provides the live camera stream.

```text
GET /api/incident
```

Returns the latest safety incident.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/vishal945vr/RAG_Project.git
```

Move into the project:

```bash
cd RAG_Project
```

---

## 2. Create Virtual Environment

Windows:

```powershell
python -m venv venv
```

Activate:

```powershell
venv\Scripts\activate
```

---

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root.

Example:

```env
GROQ_API_KEY=your_api_key_here
```

The application loads environment variables using `python-dotenv`.

### Security

Do not publish your real API key to GitHub.

Add `.env` to `.gitignore` if the repository is public.

---

# ▶️ Run the Application Locally

From the project root:

```powershell
python -m frontend.app
```

The application should start on:

```text
http://127.0.0.1:5000
```

Open the URL in your browser.

---

# 🐳 Docker

The project can also be containerized using Docker.

## Build Docker Image

```powershell
docker build -t industrial-safety-copilot .
```

## Run Container

If environment variables are supplied externally:

```powershell
docker run --env-file .env -p 5000:5000 industrial-safety-copilot
```

Then open:

```text
http://localhost:5000
```

---

# 📦 YOLO Model

The application expects a trained YOLO model:

```text
runs/detect/train-4/weights/best.pt
```

The model path can be configured using:

```env
MODEL_PATH=runs/detect/train-4/weights/best.pt
```

This makes the application easier to deploy in different environments.

---

# 📸 Visual Evidence Workflow

```text
Camera Start
     │
     ▼
Live Video
     │
     ▼
YOLO Detection
     │
     ▼
Incident Created
     │
     ▼
Evidence Snapshot
     │
     ▼
Camera Stop
     │
     ▼
Last Evidence Returned
     │
     ▼
Visual Evidence Dashboard
```

---

# 🛡️ Safety Design Principles

The application follows several important principles:

### YOLO is the evidence layer

The LLM should not change or invent detection results.

### RAG is the policy layer

Safety-policy information should come from the retrieved documents available to the system.

### LLM is the reasoning layer

The LLM interprets the incident and policy context to provide a human-readable response.

### Risk score comes from application logic

The LLM does not create a second independent risk score.

---

# 🔄 Complete Application Flow

```text
             USER
               │
               ▼
        Flask Dashboard
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
   Image Upload      Live Camera
       │                │
       └───────┬────────┘
               ▼
             YOLO
               │
               ▼
       PPE Detection
               │
               ▼
       Incident Builder
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
   Risk Analysis    Evidence
       │                │
       └───────┬────────┘
               ▼
          Hybrid RAG
               │
               ▼
       Safety Policies
               │
               ▼
             LLM
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
Safety Copilot      AI Report
```

---

# 🎯 Use Cases

Industrial Safety Copilot can be used for:

* Factory safety monitoring
* PPE compliance monitoring
* Construction site safety
* Warehouse safety
* Manufacturing environments
* Workplace incident analysis
* Safety policy assistance
* Safety inspection reporting

---

# 🔮 Future Improvements

Potential future enhancements include:

* Multi-camera monitoring
* Video-based incident detection
* Automatic incident history
* PostgreSQL / MongoDB integration
* User authentication
* Role-based dashboards
* Email/SMS alerts
* Cloud deployment
* GPU-enabled inference
* Advanced analytics
* Safety trend dashboards
* Automatic PDF incident reports
* Centralized safety policy management

---

# ⚠️ Disclaimer

Industrial Safety Copilot is an AI-assisted safety monitoring system.

Computer vision predictions and AI-generated responses may contain errors. The system should support, not replace, qualified safety professionals, established company procedures, and applicable safety requirements.

---

# 👨‍💻 Author

**Vishal Rajput**

Industrial Safety Copilot
AI + Computer Vision + RAG + LLM

---

## ⭐ Project Goal

The goal of Industrial Safety Copilot is to combine:

```text
Computer Vision
       +
Retrieval-Augmented Generation
       +
Large Language Models
       +
Industrial Safety
```

to create an intelligent assistant for workplace safety monitoring and incident analysis.

````

Isko project ke root mein **`README.md`** naam se save karo:

```text
D:\rag_project\README.md
````

Phir GitHub par push:

```powershell
git add README.md
git commit -m "Add project documentation"
git push origin main
```

Aapka GitHub project ab README ke saath properly documented ho jayega.
=======
# -Industrial-Safety-AI-Copilot
>>>>>>> afb0c8ec06681a9944c33ac1ea2a66ddc05aefb5
