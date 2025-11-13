import cv2
import face_recognition
import numpy as np
import sounddevice as sd
import winsound
import time

# ---------------------- Setup -------------------------
num = int(input("Enter number of students to register: "))
known_faces = []
known_names = []

cap = cv2.VideoCapture(0)
print("\n[INFO] Capturing faces... Look at the camera")

for i in range(num):
    name = input(f"Enter name of student {i+1}: ")
    print(f"Capturing {name}... please look straight for 5 seconds.")
    time.sleep(2)
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if encodings:
        known_faces.append(encodings[0])
        known_names.append(name)
        print(f"[INFO] Registered {name}")
    else:
        print(f"[WARN] No face detected for {name}, try again!")

print("\n✅ Registration complete.")
time.sleep(1)

# ---------------------- Confirmation Before Monitoring -------------------------
start = input("\nShould we start proctoring or not? (yes/no): ").strip().lower()
if start != "yes":
    print("❌ Proctoring cancelled by user.")
    cap.release()
    exit()
else:
    print("\n🎥 Starting Smart Proctoring...\n")
    time.sleep(1)

# ---------------------- Whisper Detection -------------------------
def detect_whisper(threshold=0.02, duration=2, fs=16000):
    """Detects whisper or sudden audio spikes"""
    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, blocking=True)
        amp = np.max(np.abs(audio))
        return amp > threshold
    except Exception as e:
        print("Audio Error:", e)
        return False

# ---------------------- Gaze Direction Estimation -------------------------
def get_head_direction(landmarks):
    """Approximate left/right gaze direction using eye & nose positions"""
    try:
        left_eye = np.mean(landmarks[0]["left_eye"], axis=0)
        right_eye = np.mean(landmarks[0]["right_eye"], axis=0)
        nose_bridge = np.mean(landmarks[0]["nose_bridge"], axis=0)

        yaw_ratio = (nose_bridge[0] - left_eye[0]) / (right_eye[0] - left_eye[0])
        if yaw_ratio < 0.35:
            return "Left"
        elif yaw_ratio > 0.65:
            return "Right"
        else:
            return "Center"
    except Exception:
        return "Center"

# ---------------------- Monitoring Loop -------------------------
last_audio_check = time.time()
fps_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Downscale for speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, faces)

    # Process each face live
    for (top, right, bottom, left), enc in zip(faces, encodings):
        matches = face_recognition.compare_faces(known_faces, enc)
        name = "Unknown"

        if True in matches:
            idx = matches.index(True)
            name = known_names[idx]
            color = (0, 255, 0)
        else:
            name = "Outsider"
            color = (0, 0, 255)
            cv2.putText(frame, "⚠️ Outsider Appeared!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            winsound.Beep(1000, 200)

        # Scale coords back
        top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Continuous gaze tracking per frame
        if name != "Outsider":
            landmarks = face_recognition.face_landmarks(rgb)
            if landmarks:
                direction = get_head_direction(landmarks)
                if direction in ["Left", "Right"]:
                    cv2.putText(frame, f"Student Unfocused ({direction})!", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    winsound.Beep(1500, 200)
                else:
                    cv2.putText(frame, "Student Focused", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Whisper Detection every 8 sec
    if time.time() - last_audio_check > 8:
        if detect_whisper():
            print("🔊 Whisper Detected!")
            winsound.Beep(1700, 300)
        last_audio_check = time.time()

    # Show FPS counter (for testing smoothness)
    fps = 1 / (time.time() - fps_time)
    fps_time = time.time()
    cv2.putText(frame, f"FPS: {int(fps)}", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Smart Proctoring Demo (Real-time)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
