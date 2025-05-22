# 🏋️ Real-Time Squat Analysis Tool (YOLOv8 + PyQt5)

This project provides a real-time squat form analyzer using **YOLOv8 pose estimation** with a **PyQt5-based GUI**. It detects key performance angles during a squat — specifically the **knee angle** and **torso angle** — and provides **visual feedback** via a traffic light system, as well as **rep counting with audio cues**.

---

## 📸 Features

- 🔍 Real-time pose detection (17 keypoints) using YOLOv8 pose model.
- ✅ Measures:
  - **Knee joint angle** (hip–knee–ankle)
  - **Torso angle** relative to vertical (shoulder–hip)
- 🟢🔴 Traffic light indicator:
  - **Green:** Legal squat (≤ 90°)
  - **Amber:** Approaching depth (91°–110°)
  - **Red:** Not deep enough (> 110°)
- 🔊 Audio beep every time a valid rep is counted
- 🖥️ Professional GUI:
  - Start/Stop buttons
  - Live webcam feed
  - On-screen metrics and feedback

---

## 💡 Best Results

- ⚠️ **Position the camera side-on to the athlete.**  
  This ensures the key angles (knee and torso) are captured accurately in 2D.

---

## 🚀 Getting Started

### 1. Clone the repo:
```bash
git clone https://github.com/your-username/squat-analyzer.git
cd squat-analyzer

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
