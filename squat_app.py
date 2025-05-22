import sys
import cv2
import numpy as np
import time
import winsound
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO


class SquatAnalyzer(QThread):
    update_frame = pyqtSignal(QImage)
    update_info = pyqtSignal(int, float, float, str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.model = YOLO('yolov8n-pose.pt').to('cuda')
        self.model.fuse()
        self.rep_count = 0
        self.in_squat = False

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return np.degrees(angle)

    def get_status_color(self, knee_angle):
        if knee_angle > 110:
            return (0, 0, 255), "RED"
        elif knee_angle > 90:
            return (0, 165, 255), "AMBER"
        else:
            return (0, 255, 0), "GREEN"

    def run(self):
        cap = cv2.VideoCapture(0)
        self.rep_count = 0
        self.in_squat = False
        prev_time = time.time()

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, device='cuda', half=True, verbose=False)
            keypoints = results[0].keypoints

            knee_angle, torso_angle, status_label = 0.0, 0.0, ""

            if keypoints.shape[0] > 0:
                kp = keypoints[0].cpu().numpy()
                try:
                    r_hip = kp[8][:2]
                    r_knee = kp[10][:2]
                    r_ankle = kp[12][:2]
                    r_shoulder = kp[6][:2]

                    knee_angle = self.calculate_angle(r_hip, r_knee, r_ankle)
                    torso_angle = self.calculate_angle(r_shoulder, r_hip, [r_hip[0], r_hip[1] + 100])
                    color, status_label = self.get_status_color(knee_angle)

                    if knee_angle <= 90 and not self.in_squat:
                        self.in_squat = True
                        self.rep_count += 1
                        winsound.Beep(1000, 150)
                    elif knee_angle > 110 and self.in_squat:
                        self.in_squat = False

                    cv2.putText(frame, f'Knee: {knee_angle:.1f}\u00b0', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f'Torso: {torso_angle:.1f}\u00b0', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f'Status: {status_label}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f'Reps: {self.rep_count}', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.circle(frame, (580, 60), 25, color, -1)
                except IndexError:
                    pass

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.update_frame.emit(qt_image)
            self.update_info.emit(self.rep_count, knee_angle, torso_angle, status_label)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class SquatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Squat Analysis Tool")
        self.setGeometry(100, 100, 700, 600)

        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid gray")

        self.rep_label = QLabel("Reps: 0")
        self.knee_label = QLabel("Knee Angle: 0.0°")
        self.torso_label = QLabel("Torso Angle: 0.0°")
        self.status_label = QLabel("Status: -")

        for label in [self.rep_label, self.knee_label, self.torso_label, self.status_label]:
            label.setStyleSheet("font-size: 16px")

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.stop_analysis)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)

        info_layout = QVBoxLayout()
        info_layout.addWidget(self.rep_label)
        info_layout.addWidget(self.knee_label)
        info_layout.addWidget(self.torso_label)
        info_layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)

        layout.addLayout(info_layout)
        layout.addLayout(buttons)
        self.setLayout(layout)

        self.worker = SquatAnalyzer()
        self.worker.update_frame.connect(self.set_image)
        self.worker.update_info.connect(self.update_metrics)

    def set_image(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def update_metrics(self, reps, knee_angle, torso_angle, status):
        self.rep_label.setText(f"Reps: {reps}")
        self.knee_label.setText(f"Knee Angle: {knee_angle:.1f}°")
        self.torso_label.setText(f"Torso Angle: {torso_angle:.1f}°")
        self.status_label.setText(f"Status: {status}")

    def start_analysis(self):
        self.worker.running = True
        self.worker.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_analysis(self):
        self.worker.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SquatApp()
    win.show()
    sys.exit(app.exec_())
