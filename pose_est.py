import cv2
import numpy as np
import time
import winsound  # Beep sound for rep feedback (Windows only)
from ultralytics import YOLO

# Load model and push to GPU
model = YOLO('yolov8n-pose.pt').to('cuda')
model.fuse()

# Angle calculation between 3 points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

# Traffic light color status
def get_status_color(knee_angle):
    if knee_angle > 110:
        return (0, 0, 255), "RED"
    elif knee_angle > 90:
        return (0, 165, 255), "AMBER"
    else:
        return (0, 255, 0), "GREEN"

# Setup webcam
cap = cv2.VideoCapture(0)
prev_time = time.time()

# Rep counter
rep_count = 0
in_squat = False  # State flag

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run pose inference
    results = model(frame, device='cuda', half=True, verbose=False)
    keypoints = results[0].keypoints

    if keypoints.shape[0] > 0:
        kp = keypoints[0].cpu().numpy()

        try:
            # Right side body joints (COCO format)
            r_hip = kp[8][:2]
            r_knee = kp[10][:2]
            r_ankle = kp[12][:2]
            r_shoulder = kp[6][:2]

            # Calculate angles
            knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            torso_angle = calculate_angle(r_shoulder, r_hip, [r_hip[0], r_hip[1] + 100])

            # Rep detection and beep
            if knee_angle <= 90 and not in_squat:
                in_squat = True
                rep_count += 1
                winsound.Beep(1000, 150)  # 1000 Hz beep for 150 ms
            elif knee_angle > 110 and in_squat:
                in_squat = False

            # Traffic light color
            color, label = get_status_color(knee_angle)

            # Draw overlay
            cv2.putText(frame, f'Knee: {knee_angle:.1f}°', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f'Torso: {torso_angle:.1f}°', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f'Status: {label}', (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f'Reps: {rep_count}', (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.circle(frame, (600, 100), 30, color, -1)

        except IndexError:
            pass

    # FPS overlay
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {fps:.1f}', (30, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)

    # Display window
    cv2.imshow("Real-Time Squat Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
