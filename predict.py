import cv2
import numpy as np
from PIL import Image
import sys

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QHBoxLayout, QVBoxLayout, QLabel
from PyQt5.QtCore import QThread, pyqtSignal
from yolo import YOLO


class DetectThread(QThread):
    result_ready = pyqtSignal(np.ndarray)

    def __init__(self, image_path):
        super(DetectThread, self).__init__()
        self.image_path = image_path

    def run(self):
        yolo = YOLO()
        img = Image.open(self.image_path)
        detected_image = yolo.detect_image(img)

        # 转换为numpy数组
        detected_image = np.array(detected_image)

        # 再转换为BGR格式
        detected_image = cv2.cvtColor(detected_image, cv2.COLOR_RGB2BGR)

        self.result_ready.emit(detected_image)


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("口罩检测")
        self.setFixedSize(2000,1000)
        layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.btn_select_image = QPushButton('选择图片')
        self.btn_detect = QPushButton('开始检测')
        btn_layout.addWidget(self.btn_select_image)
        btn_layout.addWidget(self.btn_detect)

        self.img_raw = QLabel()
        self.img_raw.setFixedSize(940,900)
        self.img_raw.setScaledContents(True)  # 图片自适应大小
        self.img_result = QLabel()
        self.img_result.setFixedSize(940,900)
        self.img_result.setScaledContents(True)  # 图片自适应大小


        img_layout = QHBoxLayout()
        img_layout.addWidget(self.img_raw)
        img_layout.addWidget(self.img_result)

        layout.addLayout(btn_layout)
        layout.addLayout(img_layout)
        self.setLayout(layout)

        self.btn_select_image.clicked.connect(self.select_image)
        self.btn_detect.clicked.connect(self.detect_image)

        self.detect_thread = None

    def select_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '打开图片', '.', '图像文件(*.jpg *.png)')
        if fname:
            self.image_path = fname
            pixmap = QPixmap(fname)
            self.img_raw.setPixmap(pixmap)

    def detect_image(self):
        if not self.image_path:
            return

        # 启动后台线程进行图像检测
        self.setWindowTitle("正在检测......")
        self.detect_thread = DetectThread(self.image_path)
        self.detect_thread.result_ready.connect(self.update_ui)
        self.detect_thread.start()

    def update_ui(self, detected_image):
        # 将numpy数组转换为QImage
        self.setWindowTitle("口罩检测")
        height, width, channel = detected_image.shape
        bytes_per_line = 3 * width
        qimage = QImage(detected_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimage)
        self.img_result.setPixmap(pixmap.scaled(self.img_result.width(), self.img_result.height()))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
