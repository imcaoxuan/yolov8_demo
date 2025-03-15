import sys
from queue import Queue

import cv2
from PySide6.QtCore import QThread, Signal, Slot, QSize, QRect
from PySide6.QtGui import QPixmap, QImage, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLabel

from infer import YOLOv8


class VideoThread(QThread):
    change_pixmap_signal = Signal(QImage)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self._run_flag = True
        self.model = 'yolov8s.onnx'
        self.conf_thres = 0.6
        self.iou_thres = 0.5

    def run(self):

        cap = cv2.VideoCapture(self.video_path)
        detection = YOLOv8(self.model, self.conf_thres, self.iou_thres)
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                img = cv2.resize(frame, (640, 640))
                output_image = detection.main(img)
                h, w, ch = output_image.shape
                bytes_per_line = ch * w
                self.change_pixmap_signal.emit(QImage(output_image.data, w, h, bytes_per_line, QImage.Format.Format_BGR888))
            else:
                break
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Yolov8s Demo")
        self.setGeometry(QRect(0, 0, 640, 640))
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        # self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        # self.media_player.setVideoOutput(self.video_widget)
        #
        self.open_button = QPushButton("选择视频")
        self.open_button.clicked.connect(self.open_file)

        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop)

        self.label = QLabel(self)

        # self.label.setPixmap()

        layout = QVBoxLayout()
        layout.addWidget(self.open_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.label, alignment=Qt.AlignmentFlag.AlignCenter)

        # layout.addWidget(self.play_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_name:
            self.detect(file_name)

    @Slot(QImage)
    def update_image(self, qt_img):
        self.label.setPixmap(QPixmap(qt_img))

    def detect(self, video_path: str):
        self.thread = VideoThread(video_path)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()


    def stop(self):
        self.thread.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec())
