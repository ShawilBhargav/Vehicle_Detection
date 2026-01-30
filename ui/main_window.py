import sys
import cv2

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal

from detector.vehicle_detector import detect_vehicles


# ---------------- WORKER THREAD ----------------

class DetectionWorker(QThread):

    frame_ready = pyqtSignal(object, int)

    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        self.running = True

    def run(self):

        while self.running:

            ret, frame = self.cap.read()

            if not ret:
                break

            processed_frame, total_count = detect_vehicles(frame)

            self.frame_ready.emit(processed_frame, total_count)

    def stop(self):
        self.running = False


# ---------------- MAIN WINDOW ----------------

class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.cap = cv2.VideoCapture("../videos/traffic.mp4")
        self.worker = None

        self.setWindowTitle("Vehicle Detection System")
        self.setGeometry(200, 100, 1000, 600)

        self.init_ui()

    def init_ui(self):

        # Video display
        self.video_label = QLabel()
        self.video_label.setFixedSize(700, 450)
        self.video_label.setStyleSheet("border: 2px solid black")

        # Counter label
        self.total_label = QLabel("Vehicles: 0")

        # Buttons
        self.start_btn = QPushButton("Start Detection")
        self.stop_btn = QPushButton("Stop")

        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)

        # Layouts
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)

        side_layout = QVBoxLayout()
        side_layout.addWidget(self.total_label)

        top_layout = QHBoxLayout()
        top_layout.addLayout(video_layout)
        top_layout.addLayout(side_layout)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def start_detection(self):

        if self.cap:

            self.worker = DetectionWorker(self.cap)
            self.worker.frame_ready.connect(self.update_gui)
            self.worker.start()

    def stop_detection(self):

        if self.worker:
            self.worker.stop()

    def update_gui(self, frame, total_count):

        self.total_label.setText(f"Vehicles: {total_count}")

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb_image.shape

        qt_image = QImage(
            rgb_image.data,
            w,
            h,
            ch * w,
            QImage.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(qt_image)

        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.width(),
                self.video_label.height()
            )
        )


# ---------------- APP START ----------------

if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())