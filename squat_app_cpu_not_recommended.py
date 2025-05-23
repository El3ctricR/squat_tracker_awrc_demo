import sys
import cv2
import numpy as np
import winsound
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QMessageBox, QGroupBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO

# === Constants ===
MODEL_PATH = r'model path (download from ultralytics site)'
LOGO_PATH = r'logo path (optional)'

SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (5, 6), (11, 12),
    (5, 11), (6, 12)
]
COLORS = [
    (255, 0, 0), (255, 0, 0),
    (0, 255, 0), (0, 255, 0),
    (255, 0, 255), (255, 0, 255),
    (0, 165, 255), (0, 165, 255),
    (255, 255, 255), (255, 255, 255),
    (255, 255, 0), (255, 255, 0)
]


class SquatAnalyzer(QThread):
    update_frame = pyqtSignal(QImage)
    update_info = pyqtSignal(int, float, float, str)

    # Load model once on CPU
    try:
        model = YOLO(MODEL_PATH).to('cpu')
        model.fuse()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    def __init__(self):
        super().__init__()
        self.running = False
        self.rep_count = 0
        self.in_squat = False
        self.cap = None
        self.logo = self.load_logo(LOGO_PATH)

    @staticmethod
    def load_logo(path):
        try:
            logo = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if logo is not None and logo.shape[2] == 3:
                logo = cv2.cvtColor(logo, cv2.COLOR_BGR2BGRA)
            return logo
        except:
            return None

    @staticmethod
    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0:
            return 0.0
        cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return np.degrees(angle)

    @staticmethod
    def get_status_color(knee_angle):
        if knee_angle > 110:
            return (0, 0, 255), "RED"
        elif knee_angle > 90:
            return (0, 165, 255), "AMBER"
        else:
            return (0, 255, 0), "GREEN"

    def run(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        self.rep_count = 0
        self.in_squat = False

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(rgb_frame, device='cpu', verbose=False)
            keypoints = results[0].keypoints
            knee_angle, torso_angle, status_label = 0.0, 0.0, ""

            if keypoints is not None and keypoints.data.shape[1] >= 17:
                conf = keypoints.conf[0].numpy()
                kp = keypoints.data[0].numpy()

                def is_valid(index, threshold=0.3):
                    return conf[index] > threshold and not np.allclose(kp[index][:2], [0, 0])

                for (i, j), color in zip(SKELETON, COLORS):
                    if is_valid(i) and is_valid(j):
                        pt1 = tuple(map(int, kp[i][:2]))
                        pt2 = tuple(map(int, kp[j][:2]))
                        cv2.line(frame, pt1, pt2, color, 2)

                required_indices = [6, 12, 14, 16]
                if all(is_valid(i) for i in required_indices):
                    r_shoulder, r_hip, r_knee, r_ankle = (kp[i][:2] for i in required_indices)
                    knee_angle = self.calculate_angle(r_hip, r_knee, r_ankle)
                    torso_angle = self.calculate_angle(r_hip, r_shoulder, r_knee)
                    color, status_label = self.get_status_color(knee_angle)

                    if knee_angle <= 90 and not self.in_squat:
                        self.in_squat = True
                        self.rep_count += 1
                        winsound.Beep(1000, 150)
                    elif knee_angle > 110 and self.in_squat:
                        self.in_squat = False

                    cv2.putText(frame, f'Knee: {knee_angle:.1f}°', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f'Torso: {torso_angle:.1f}°', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f'Status: {status_label}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f'Reps: {self.rep_count}', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.circle(frame, (580, 60), 25, color, -1)

            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = display_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.update_frame.emit(qt_image)
            self.update_info.emit(self.rep_count, knee_angle, torso_angle, status_label)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def stop(self):
        self.running = False
        self.wait()


class SquatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Squat Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: white; color: black;")

        title_label = QLabel("Squat Analysis Tool")
        title_label.setObjectName("TitleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; padding: 10px; color: #333333;")

        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #8be9fd; border-radius: 8px;")

        self.rep_label = QLabel("Reps: 0")
        self.knee_label = QLabel("Knee Angle: 0.0°")
        self.torso_label = QLabel("Torso Angle: 0.0°")
        self.status_label = QLabel("Status: -")

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.stop_analysis)

        info_layout = QVBoxLayout()
        for lbl in [self.rep_label, self.knee_label, self.torso_label, self.status_label]:
            info_layout.addWidget(lbl)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(title_label)
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(info_layout)
        main_layout.addLayout(control_layout)
        self.setLayout(main_layout)

        self.worker = SquatAnalyzer()
        self.worker.update_frame.connect(self.set_image)
        self.worker.update_info.connect(self.update_metrics)

        QMessageBox.information(self, "Setup Guidance", "Please stand side-on with your RIGHT side facing the camera.")

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
