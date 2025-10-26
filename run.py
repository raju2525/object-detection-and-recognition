import sys
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QMessageBox, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap

from PyQt5.QtCore import QTimer, Qt  
# Load YOLOv5 model (use CPU by default)
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("Model loaded successfully!")

class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integrated Recognition & Detection using CNN (YOLOv5)")
        self.setGeometry(200, 100, 1000, 700)
        
        self.image_label = QLabel("Detection Preview", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid gray; background-color: #202020; color: white;")
        
        # Buttons
        self.upload_img_btn = QPushButton("Upload Image")
        self.upload_video_btn = QPushButton("Upload Video")
        self.webcam_btn = QPushButton("Webcam Feed")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        # Layouts
        hbox = QHBoxLayout()
        hbox.addWidget(self.upload_img_btn)
        hbox.addWidget(self.upload_video_btn)
        hbox.addWidget(self.webcam_btn)
        hbox.addWidget(self.stop_btn)

        layout = QVBoxLayout()
        layout.addLayout(hbox)
        layout.addWidget(self.image_label)
        self.setLayout(layout)

        # Signals
        self.upload_img_btn.clicked.connect(self.load_image)
        self.upload_video_btn.clicked.connect(self.load_video)
        self.webcam_btn.clicked.connect(self.start_webcam)
        self.stop_btn.clicked.connect(self.stop_feed)

        # Variables
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None

    def detect_and_display(self, frame):
        results = model(frame)
        df = results.pandas().xyxy[0]
        for _, row in df.iterrows():
            xmin, ymin, xmax, ymax, conf, cls, name = row
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return frame

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                QMessageBox.warning(self, "Error", "Cannot read image.")
                return
            result = self.detect_and_display(img)
            self.display_image(result)

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Error", "Cannot open video file.")
                return
            self.timer.start(30)
            self.stop_btn.setEnabled(True)

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Error", "Cannot access webcam.")
            return
        self.timer.start(30)
        self.stop_btn.setEnabled(True)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_feed()
            return
        frame = self.detect_and_display(frame)
        self.display_image(frame)

    def stop_feed(self):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        self.stop_btn.setEnabled(False)

    def display_image(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.image_label.width(), self.image_label.height(), aspectRatioMode=1))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())
