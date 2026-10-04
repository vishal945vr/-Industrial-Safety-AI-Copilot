import os
import sys

# ============================================================
# PROJECT ROOT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import os
import cv2
import time
import uuid
import threading

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    Response,
    send_from_directory
)

from werkzeug.utils import secure_filename
from ultralytics import YOLO

# ============================================================
# YOLO FUNCTIONS
# ============================================================

from vision.yolo_code import (
    MODEL_PATH,
    run_detection,
    build_incident,
    overlay_risk_info,
    save_incident,
    maybe_save_snapshot
)

# ============================================================
# RAG / LLM FUNCTIONS
# ============================================================

from scr.retrive import (
    generate_safety_report,
    ask_safety_copilot
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.getcwd()

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "predict"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


# ============================================================
# ALLOWED FILES
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
    "mp4",
    "avi",
    "mov",
    "mkv"
}


def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# LOAD YOLO MODEL ONCE
# ============================================================

print("\n" + "=" * 60)
print(" INDUSTRIAL SAFETY COPILOT")
print("=" * 60)

print("\nLoading YOLO model...")

try:

    model = YOLO(MODEL_PATH)

    print("YOLO model loaded successfully.")

except Exception as e:

    print("ERROR loading YOLO model:")
    print(e)

    model = None


# ============================================================
# GLOBAL STATE
# ============================================================

latest_incident = None

latest_detection = {
    "detected": {},
    "confidences": {}
}

latest_image = None


# ============================================================
# CAMERA STATE
# ============================================================

camera = None

camera_running = False

camera_lock = threading.Lock()

camera_thread = None


# ============================================================
# CAMERA CONFIG
# ============================================================

CAMERA_INDEX = 0

CAMERA_ID = "CAM-01"

# Run YOLO every N frames.
# 1 = every frame
# 2 = every second frame
# 3 = every third frame
DETECTION_INTERVAL = 3


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template("index.html")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "message": "Industrial Safety Copilot is running."
    })


# ============================================================
# IMAGE ANALYSIS
# ============================================================

@app.route("/api/analyze", methods=["POST"])
def analyze():

    global latest_incident
    global latest_detection
    global latest_image

    try:

        if model is None:

            return jsonify({
                "success": False,
                "error": "YOLO model is not loaded."
            }), 500


        # ----------------------------------------------------
        # CHECK FILE
        # ----------------------------------------------------

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "error": "No file uploaded."
            }), 400


        file = request.files["file"]


        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "No file selected."
            }), 400


        if not allowed_file(file.filename):

            return jsonify({
                "success": False,
                "error": "Unsupported file type."
            }), 400


        # ----------------------------------------------------
        # SAVE UPLOAD
        # ----------------------------------------------------

        original_name = secure_filename(
            file.filename
        )

        extension = original_name.rsplit(
            ".",
            1
        )[1].lower()

        unique_name = (
            f"{uuid.uuid4().hex[:12]}."
            f"{extension}"
        )

        upload_path = os.path.join(
            UPLOAD_FOLDER,
            unique_name
        )

        file.save(upload_path)


        # ====================================================
        # IMAGE
        # ====================================================

        if extension in {
            "jpg",
            "jpeg",
            "png",
            "webp"
        }:

            frame = cv2.imread(upload_path)


            if frame is None:

                return jsonify({
                    "success": False,
                    "error": "Could not read uploaded image."
                }), 400


            # ------------------------------------------------
            # YOLO
            # ------------------------------------------------

            annotated_frame, detected, confidences = (
                run_detection(
                    model,
                    frame
                )
            )


            # ------------------------------------------------
            # SAVE DETECTION STATE
            # ------------------------------------------------

            latest_detection = {
                "detected": detected,
                "confidences": confidences
            }


            # ------------------------------------------------
            # CREATE INCIDENT
            # ------------------------------------------------

            incident_id = (
                "INC-" +
                uuid.uuid4().hex[:6].upper()
            )


            incident = build_incident(
                detected,
                confidences,
                camera_id="UPLOAD",
                incident_id=incident_id
            )


            # ------------------------------------------------
            # RISK OVERLAY
            # ------------------------------------------------

            annotated_frame = overlay_risk_info(
                annotated_frame,
                incident
            )


            # ------------------------------------------------
            # OUTPUT IMAGE
            # ------------------------------------------------

            output_filename = (
                f"result_{incident_id}.jpg"
            )

            output_path = os.path.join(
                OUTPUT_FOLDER,
                output_filename
            )


            cv2.imwrite(
                output_path,
                annotated_frame
            )


            # ------------------------------------------------
            # SAVE INCIDENT JSON
            # ------------------------------------------------

            try:

                incident_json_path = os.path.join(
                    OUTPUT_FOLDER,
                    f"{incident_id}.json"
                )

                save_incident(
                    incident,
                    incident_json_path
                )

            except Exception as e:

                print(
                    "Incident JSON save error:",
                    e
                )


            # ------------------------------------------------
            # SAVE HIGH/CRITICAL SNAPSHOT
            # ------------------------------------------------

            try:

                maybe_save_snapshot(
                    annotated_frame,
                    incident
                )

            except Exception as e:

                print(
                    "Snapshot save error:",
                    e
                )


            # ------------------------------------------------
            # UPDATE GLOBAL INCIDENT
            # ------------------------------------------------

            latest_incident = incident

            latest_image = output_filename


            # ------------------------------------------------
            # RESPONSE
            # ------------------------------------------------

            return jsonify({

                "success": True,

                "incident": incident,

                "image_url":
                    "/result/" +
                    output_filename

            })


        # ====================================================
        # VIDEO
        # ====================================================

        elif extension in {
            "mp4",
            "avi",
            "mov",
            "mkv"
        }:

            return jsonify({

                "success": True,

                "message":
                    "Video uploaded successfully. "
                    "Use the live camera mode for real-time detection.",

                "filename":
                    unique_name

            })


    except Exception as e:

        print("\nIMAGE ANALYSIS ERROR:")
        print(e)

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# SAFETY COPILOT
# ============================================================

@app.route("/api/ask", methods=["POST"])
def ask_safety_copilot_api():

    global latest_incident

    try:

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({
                "success": False,
                "error": "Invalid request."
            }), 400


        question = data.get(
            "question",
            ""
        ).strip()


        if not question:

            return jsonify({
                "success": False,
                "error": "Please enter a question."
            }), 400


        if latest_incident is None:

            return jsonify({
                "success": False,
                "error":
                    "Please analyze an image or "
                    "start the live camera first."
            }), 400


        print("\n" + "=" * 60)
        print("SAFETY COPILOT")
        print("=" * 60)

        print("Question:")
        print(question)


        # ----------------------------------------------------
        # RAG + LLM
        # ----------------------------------------------------

        answer = ask_safety_copilot(
            question,
            latest_incident
        )


        return jsonify({

            "success": True,

            "answer": answer

        })


    except Exception as e:

        print("\nSAFETY COPILOT ERROR:")
        print(e)

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# AI REPORT
# ============================================================

@app.route("/api/report", methods=["GET"])
def generate_ai_report():

    global latest_incident

    try:

        if latest_incident is None:

            return jsonify({

                "success": False,

                "error":
                    "No incident available. "
                    "Analyze an image or start the camera first."

            }), 400


        print("\n" + "=" * 60)
        print("GENERATING AI SAFETY REPORT")
        print("=" * 60)


        # ----------------------------------------------------
        # RAG + LLM
        # ----------------------------------------------------

        report = generate_safety_report(
            latest_incident
        )


        return jsonify({

            "success": True,

            "report": report,

            "incident": latest_incident

        })


    except Exception as e:

        print("\nREPORT GENERATION ERROR:")
        print(e)

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# RESULT IMAGE
# ============================================================

@app.route("/result/<filename>")
def result_file(filename):

    return send_from_directory(
        OUTPUT_FOLDER,
        filename
    )


# ============================================================
# UPLOADED FILE
# ============================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ============================================================
# LATEST INCIDENT
# ============================================================

@app.route("/api/incident", methods=["GET"])
def get_incident():

    if latest_incident is None:

        return jsonify({

            "success": False,

            "message":
                "No incident available."

        })


    return jsonify({

        "success": True,

        "incident": latest_incident

    })


# ============================================================
# LATEST DETECTION
# ============================================================

@app.route("/api/detection", methods=["GET"])
def get_detection():

    return jsonify({

        "success": True,

        "detection": latest_detection

    })


# ============================================================
# CAMERA START
# ============================================================

@app.route("/api/camera/start", methods=["GET", "POST"])
def start_camera():

    global camera_running
    global camera_thread

    with camera_lock:

        if camera_running:

            return jsonify({

                "success": True,

                "message":
                    "Camera is already running."

            })


        camera_running = True


        # Start background camera thread
        camera_thread = threading.Thread(
            target=camera_worker,
            daemon=True
        )

        camera_thread.start()


    return jsonify({

        "success": True,

        "message":
            "Camera started."

    })


# ============================================================
# CAMERA STOP
# ============================================================

@app.route("/api/camera/stop", methods=["GET", "POST"])
def stop_camera():

    global camera_running
    global camera

    with camera_lock:

        camera_running = False


        if camera is not None:

            try:
                camera.release()
            except Exception:
                pass

            camera = None


    return jsonify({

        "success": True,

        "message":
            "Camera stopped."

    })


# ============================================================
# CAMERA WORKER
# ============================================================

def camera_worker():

    global camera

    print("\nStarting webcam...")

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    camera = cap


    if not cap.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        global camera_running

        camera_running = False

        return


    # --------------------------------------------------------
    # Camera settings
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )


    print(
        "Webcam started successfully."
    )


    frame_count = 0


    while camera_running:

        success, frame = cap.read()


        if not success:

            print(
                "Camera frame read failed."
            )

            break


        frame_count += 1


        # ----------------------------------------------------
        # YOLO detection
        # ----------------------------------------------------

        if frame_count % DETECTION_INTERVAL == 0:

            try:

                process_camera_frame(
                    frame,
                    frame_count
                )

            except Exception as e:

                print(
                    "Camera detection error:",
                    e
                )


        # Small delay prevents CPU overload
        time.sleep(0.001)


    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()

    camera = None

    camera_running = False


    print(
        "Webcam stopped."
    )


# ============================================================
# PROCESS CAMERA FRAME
# ============================================================

def process_camera_frame(
    frame,
    frame_number
):

    global latest_incident
    global latest_detection
    global latest_image


    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    annotated_frame, detected, confidences = (
        run_detection(
            model,
            frame
        )
    )


    # --------------------------------------------------------
    # UPDATE DETECTION
    # --------------------------------------------------------

    latest_detection = {

        "detected":
            detected,

        "confidences":
            confidences

    }


    # --------------------------------------------------------
    # INCIDENT
    # --------------------------------------------------------

    incident = build_incident(

        detected,

        confidences,

        camera_id=CAMERA_ID,

        incident_id=
            f"INC-LIVE-{frame_number:06d}"

    )


    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    annotated_frame = overlay_risk_info(

        annotated_frame,

        incident

    )


    # --------------------------------------------------------
    # SAVE SNAPSHOT
    # --------------------------------------------------------

    try:

        maybe_save_snapshot(
            annotated_frame,
            incident
        )

    except Exception as e:

        print(
            "Live snapshot error:",
            e
        )


    # --------------------------------------------------------
    # UPDATE GLOBAL INCIDENT
    # --------------------------------------------------------

    latest_incident = incident


    # --------------------------------------------------------
    # SAVE LATEST CAMERA FRAME
    # --------------------------------------------------------

    latest_image = annotated_frame


# ============================================================
# LIVE VIDEO STREAM
# ============================================================

def generate_camera():

    global camera_running
    global camera

    # --------------------------------------------------------
    # Open camera if not already running
    # --------------------------------------------------------

    if not camera_running:

        with camera_lock:

            camera_running = True

            camera_thread = threading.Thread(
                target=camera_worker,
                daemon=True
            )

            camera_thread.start()


    # --------------------------------------------------------
    # Wait for camera
    # --------------------------------------------------------

    while camera is None and camera_running:

        time.sleep(0.05)


    if camera is None:

        return


    # --------------------------------------------------------
    # Streaming loop
    # --------------------------------------------------------

    while camera_running:

        try:

            success, frame = camera.read()


            if not success:

                time.sleep(0.05)

                continue


            # ------------------------------------------------
            # IMPORTANT:
            #
            # camera_worker also reads frames.
            #
            # To avoid two readers fighting over the webcam,
            # this stream reads the latest frame from the
            # camera itself.
            #
            # For stable operation, use a shared-frame system.
            # ------------------------------------------------

            annotated_frame, detected, confidences = (
                run_detection(
                    model,
                    frame
                )
            )


            incident = build_incident(

                detected,

                confidences,

                camera_id=CAMERA_ID,

                incident_id=
                    f"INC-LIVE-{int(time.time() * 1000)}"

            )


            annotated_frame = overlay_risk_info(

                annotated_frame,

                incident

            )


            # Update dashboard state
            global latest_incident
            global latest_detection

            latest_incident = incident

            latest_detection = {

                "detected":
                    detected,

                "confidences":
                    confidences

            }


            # JPEG encode
            ret, buffer = cv2.imencode(
                ".jpg",
                annotated_frame
            )


            if not ret:

                continue


            frame_bytes = buffer.tobytes()


            yield (

                b"--frame\r\n"

                b"Content-Type: image/jpeg\r\n\r\n"

                + frame_bytes +

                b"\r\n"

            )


        except GeneratorExit:

            break


        except Exception as e:

            print(
                "Video stream error:",
                e
            )

            break


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(

        generate_camera(),

        mimetype=
            "multipart/x-mixed-replace; boundary=frame"

    )



# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print(
        " INDUSTRIAL SAFETY COPILOT"
    )

    print("=" * 60)

    print(
        "\nServer:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("\n")


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True,

        threaded=True,

        use_reloader=False

    )