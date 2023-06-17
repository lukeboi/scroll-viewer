import glob
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QLabel
from PyQt5.QtGui import QImage, QPixmap, QPainter, QKeyEvent
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QKeyEvent
from PyQt5.QtCore import Qt, QRect
import glob
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QKeyEvent, QCursor
from PyQt5.QtCore import Qt, QRect, QPoint
import numpy as np

        # self.image_files = glob.glob('D:\\scroll_2_88_first_50\\s2_first50\\twoJpg\\*.jpg')

z_step = 50
segmentations = []

class ImageView(QGraphicsView):
    def __init__(self, loader, parent=None):
        super(ImageView, self).__init__(parent)
        self.setMouseTracking(True)
        self.loader = loader

    def mouseMoveEvent(self, event):
        x = event.x() + self.loader.crop_rect.x()
        y = event.y() + self.loader.crop_rect.y()
        z = self.loader.image_index
        self.loader.mouse_label.setText(f'Mouse position: x={x}, y={y}, z={z}')
        super().mouseMoveEvent(event)

        if self.loader.set_center:
            self.loader.scroll_center = (x, y)
            print(f"Scroll center = {self.loader.scroll_center}")
            self.loader.set_center = False

        if self.loader.paste_cursor:
            painter = QPainter(self.loader.pixmap)
            painter.drawImage(QPoint(x, y), self.loader.cursor_img)
            painter.end()
            self.loader.paste_cursor = False
            self.loader.load_image()
        
        if self.loader.place_segmentation:

            cutoffPlane = np.array([self.loader.scroll_center[0]-x, self.loader.scroll_center[1]-y,0])
            cutoffPlane = cutoffPlane / np.linalg.norm(cutoffPlane)

            segmentations.append(
                f"http://localhost:5001/volume?filename=scroll1&size={200},{200},{100}&origin={y * 1 - 100},{x * 1 - 100},{z - 50}&threshold=120&applySobel=true&cutoffPlane={-cutoffPlane[0]},{-cutoffPlane[1]},{cutoffPlane[2]}"
            )

            print("SEGMENTATIONS:", segmentations)

            self.loader.place_segmentation = False

class ImageLoader(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(ImageLoader, self).__init__(*args, **kwargs)

        self.image_files = glob.glob('S:\\scroll1\\oneJpgs\\*.jpg')
        self.image_index = 0

        self.scene = QGraphicsScene(self)
        self.view = ImageView(self, self)
        self.view.setScene(self.scene)

        self.crop_rect = QRect(0, 0, 1500, 1000)

        # set up mouse coordinate label
        self.mouse_label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addWidget(self.mouse_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.cursor_img = QImage("cursor.png")  # replace with your cursor image
        self.view.setCursor(QCursor(QPixmap.fromImage(self.cursor_img)))

        self.paste_cursor = False

        self.scroll_center = (0,0)
        self.set_center = False

        self.place_segmentation = False

        self.load_image()

    def load_image(self):
        self.image = QImage(self.image_files[self.image_index])
        self.pixmap = QPixmap(self.image)
        self.scene.clear()
        self.scene.addPixmap(self.pixmap.copy(self.crop_rect))

    def update_pixmap(self):
        # self.pixmap = QPixmap(self.image)
        self.scene.clear()
        self.scene.addPixmap(self.pixmap.copy(self.crop_rect))

    def keyPressEvent(self, event: QKeyEvent):
        super().keyPressEvent(event)
        if event.key() == Qt.Key_Q:
            self.image_index = (self.image_index - z_step) % len(self.image_files)
            self.load_image()
        elif event.key() == Qt.Key_E:
            self.image_index = (self.image_index + z_step) % len(self.image_files)
            self.load_image()
        elif event.key() == Qt.Key_W:
            self.crop_rect.translate(0, -100)
            self.load_image()
        elif event.key() == Qt.Key_A:
            self.crop_rect.translate(-100, 0)
            self.load_image()
        elif event.key() == Qt.Key_S:
            self.crop_rect.translate(0, 100)
            self.load_image()
        elif event.key() == Qt.Key_D:
            self.crop_rect.translate(100, 0)
            self.load_image()
        elif event.key() == Qt.Key_Space:
            self.paste_cursor = True
            self.place_segmentation = True
        elif event.key() == Qt.Key_C:
            self.set_center = True

if __name__ == '__main__':
    app = QApplication(sys.argv)

    win = ImageLoader()
    win.show()

    sys.exit(app.exec_())