
import cv2

def camera():

    # Open camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera could not be opened")
        return

    print("✅ Camera started")
    print("Press Q to exit")

    while True:

        # Read camera frame
        ret, frame = cap.read()

        if not ret:
            print("❌ Failed to read camera frame")
            break

        # Display frame
        cv2.imshow("Industrial Safety Camera", frame)

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release camera AFTER loop
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera()




    

    