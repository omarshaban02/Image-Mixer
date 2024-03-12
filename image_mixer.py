import cv2
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
from PyQt5.uic import loadUiType
import cv2
import matplotlib.pyplot as plt
import logging
from threading import Thread

mixer_logger = logging.getLogger("image_mixer.py")


class ViewOriginal(QGraphicsView):
    imageSelected = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = None
        self.original_pixmap = None
        self.path = self.path()
        self.image_viewer = None  # ImageViewer object
        mixer_logger.info("Initialize ViewOriginal class")

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.openImageDialog()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.image_viewer:
            # You can capture the mouse position during the drag here
            brightness = abs(self.mapToScene(event.pos()).x() / 1000) * 2
            contrast = (self.mapToScene(event.pos()).y() / 500) * 255
            if 1 >= brightness >= 0:
                self.image_viewer.image_brightness = brightness
            if 1 >= contrast >= 0:
                self.image_viewer.image_contrast = contrast
            image_bytes = self.image_viewer.gray_scale_image_bytes()
            self.original_pixmap.loadFromData(image_bytes.tobytes())

            if not self.pixmap_item:
                self.pixmap_item = QGraphicsPixmapItem(self.original_pixmap)
                self.scene.addItem(self.pixmap_item)

            else:
                self.pixmap_item.setPixmap(self.original_pixmap)

    def openImageDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                  "Image Files (*.png *.jpg *.bmp *.jpeg *.tif *.tiff);;All Files (*)",
                                                  options=options)
        if fileName:
            self.loadImage(fileName)
            self.imageSelected.emit(fileName)  # Emit the filename
            mixer_logger.info(f"The user open {fileName}")

            return fileName

    def path(self):
        path = self.openImageDialog
        return path

    def loadImage(self, filename):
        # self.original_pixmap = QPixmap(filename)
        self.original_pixmap = QPixmap()
        self.image_viewer = Image(filename)
        image_bytes = self.image_viewer.gray_scale_image_bytes()

        self.original_pixmap.loadFromData(image_bytes.tobytes())

        if not self.pixmap_item:
            self.pixmap_item = QGraphicsPixmapItem(self.original_pixmap)
            self.scene.addItem(self.pixmap_item)

        else:
            self.pixmap_item.setPixmap(self.original_pixmap)


class ViewWeight(QGraphicsView):
    def __init__(self):
        super().__init__()

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.image_viewer = None  # image_viewer
        self.qt_image = None
        self.current_pixmap = None
        self.resized_photo = None
        self.current_pixmap_item = None
        self._current_state = 'm'
        self._current_image = None
        self._current_clipped_image = None
        self._weight = 1
        self._is_outer_region = False

        self.drawing_rectangle = False
        self.rectangle_item = None
        self.start_point = None
        self._rect_x_limits = None
        self._rect_y_limits = None
        self._is_captured = False
        self.weights_dict = {}
        mixer_logger.info("Initialize VeiwWeight class")

    @property
    def is_outer_region(self):
        return self._is_outer_region

    @is_outer_region.setter
    def is_outer_region(self, value):
        self._is_outer_region = value

    @property
    def is_captured(self):
        return self._is_captured

    @is_captured.setter
    def is_captured(self, value):
        self._is_captured = value

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    @property
    def rect_x_limits(self):
        return self._rect_x_limits

    @rect_x_limits.setter
    def rect_x_limits(self, value):
        self._rect_x_limits = value

    @property
    def rect_y_limits(self):
        return self._rect_y_limits

    @rect_y_limits.setter
    def rect_y_limits(self, value):
        self._rect_y_limits = value

    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, value):
        self.weights_dict = {
            'm': self.image_viewer.image_magnitude,
            'p': self.image_viewer.image_phase,
            'r': self.image_viewer.image_real_part,
            'i': self.image_viewer.image_imaginary_part
        }
        self._current_state = value
        self.current_image = self.weights_dict[value]
        # if value == 'm':
        #     self.current_image = self.image_viewer.image_magnitude
        # elif value == 'p':
        #     self.current_image = self.image_viewer.image_phase
        # elif value == 'r':
        #     self.current_image = self.image_viewer.image_real_part
        # elif value == 'i':
        #     self.current_image = self.image_viewer.image_imaginary_part
        # else:
        #     mixer_logger.error("Invalid weight state")

        self.qt_image = self.convert_cv_to_qt((20 * np.log10(self.current_image)).astype(np.uint8))
        self.current_pixmap = QPixmap(self.qt_image)
        if self.current_pixmap_item:
            self.current_pixmap_item.setPixmap(self.current_pixmap)
        else:
            self.current_pixmap_item = QGraphicsPixmapItem(self.current_pixmap)
            self.scene.addItem(self.current_pixmap_item)

    @property
    def current_image(self):
        return self._current_image

    @current_image.setter
    def current_image(self, value):
        self._current_image = value

    @property
    def current_clipped_image(self):
        x_i = self.rect_x_limits[0]
        x_f = self.rect_x_limits[1]
        y_i = self.rect_y_limits[0]
        y_f = self.rect_y_limits[1]
        if self.is_outer_region:
            clipped_image = self.current_image.copy()
            clipped_image[y_i:y_f, x_i:x_f] = np.zeros_like(self.current_image[y_i:y_f, x_i:x_f])
            mixer_logger.info("Outer clipped image done")
        else:
            clipped_image = np.zeros_like(self.current_image)
            clipped_image[y_i:y_f, x_i:x_f] = self.current_image[y_i:y_f, x_i:x_f]
            mixer_logger.info("Inner clipped image done")
        self._current_clipped_image = clipped_image

        return self._current_clipped_image

    def convert_cv_to_qt(self, cv_image):
        height, width = cv_image.shape
        bytes_per_line = width
        qt_image = QImage(cv_image.data.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qt_image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.current_state != '':
                self.start_point = self.mapToScene(event.pos())
                self.drawing_rectangle = True
                self.is_captured = True
        elif event.button() == Qt.RightButton:
            self.is_captured = False
            self.drawing_rectangle = False
            self.start_point = None
            self.scene.removeItem(self.rectangle_item)
            self.rectangle_item = None

    def mouseMoveEvent(self, event):
        if event.button() != Qt.LeftButton:
            if self.drawing_rectangle:
                current_point = self.mapToScene(event.pos())
                rect = QRectF(self.start_point, current_point).normalized()
                self.rect_x_limits = (int(self.start_point.x()), int(current_point.x()))
                self.rect_y_limits = (int(self.start_point.y()), int(current_point.y()))

                if not self.rectangle_item:
                    self.rectangle_item = QGraphicsRectItem(rect)
                    pen = QPen(QColor(Qt.white), 2,
                               Qt.DashLine)  # You can use Qt.DashLine or Qt.DotLine for a dashed or dotted line
                    self.rectangle_item.setPen(pen)
                    self.scene.addItem(self.rectangle_item)
                else:
                    self.rectangle_item.setRect(rect)
        else:
            print("Mouse position during drag:", self.mapToScene(event.pos()))

    def mouseReleaseEvent(self, event):
        if self.drawing_rectangle:
            self.drawing_rectangle = False
            self.start_point = None


class Image:
    def __init__(self, path):
        self._original_image = cv2.resize(cv2.imread(f'{path}'), (320, 170), cv2.INTER_LINEAR)
        self._gray_scale_image = cv2.cvtColor(self._original_image, cv2.COLOR_BGR2GRAY)
        self._image_size = None
        self._image_brightness = 1  # The common range for alpha is from 0.0 to 2.0. (1 ==> no change)
        self._image_contrast = 0  # The common range for beta is from  -255 to 255. (0 ==> no change)
        self._image_fft = np.fft.fft2(self._gray_scale_image)
        self._image_fft_shift = np.fft.fftshift(self.image_fft)
        self._image_ifft = None
        self._image_magnitude = None
        self._image_phase = None
        self._image_real_part = None
        self._image_imaginary_part = None
        mixer_logger.info("Initialize ImageMixer class")

    @property
    def original_image(self):
        return self._original_image

    @property
    def gray_scale_image(self):
        return self._gray_scale_image

    @property
    def image_size(self):
        self._image_size = self._original_image.shape
        return self._image_size[:2]

    @image_size.setter
    def image_size(self, value):
        self._original_image = cv2.resize(self._original_image, value)
        self._image_size = value

    @property
    def image_contrast(self):
        return self._image_contrast

    @image_contrast.setter
    def image_contrast(self, value):
        if -255 <= value <= 255:
            self._image_contrast = value
            self._gray_scale_image = cv2.cvtColor(cv2.convertScaleAbs(self._original_image, alpha=self.image_brightness,
                                                                      beta=self.image_contrast), cv2.COLOR_BGR2GRAY)

    @property
    def image_brightness(self):
        return self._image_brightness

    @image_brightness.setter
    def image_brightness(self, value):
        if 0 <= value <= 2:
            self._image_brightness = value
            self._gray_scale_image = cv2.cvtColor(cv2.convertScaleAbs(self._original_image, alpha=self.image_brightness,
                                                                      beta=self.image_contrast), cv2.COLOR_BGR2GRAY)

    @property
    def image_fft(self):
        return self._image_fft

    @property
    def image_fft_shift(self):
        return self._image_fft_shift

    @image_fft_shift.setter
    def image_fft_shift(self, value):
        pass

    @property
    def image_magnitude(self):
        self._image_magnitude = np.abs(self.image_fft_shift)
        return self._image_magnitude

    @property
    def image_phase(self):
        self._image_phase = np.angle(self._image_fft_shift)
        return self._image_phase

    @property
    def image_real_part(self):
        self._image_real_part = np.real(self._image_fft_shift)
        return self._image_real_part

    @property
    def image_imaginary_part(self):
        self._image_imaginary_part = np.imag(self._image_fft_shift)
        return self._image_imaginary_part

    def gray_scale_image_bytes(self):
        _, data = cv2.imencode(img=self.gray_scale_image, ext='.jpeg')
        return data


class ImageMixer(Thread):
    def __init__(self, viewWeights: list[ViewWeight]):
        Thread.__init__(self)
        self._input_images = viewWeights
        self._mixed_fft = None
        self._mixed_image = None
        self._mode = 'mp'  # magnitude/phase or real/imaginary (mp|ri)
        self.run()

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

    @property
    def input_images(self):
        return self._input_images

    @input_images.setter
    def input_images(self, value):
        self._input_images = value

    @property
    def mixed_fft(self):
        return self._mixed_fft

    @mixed_fft.setter
    def mixed_fft(self, value):
        self._mixed_fft = value

    @property
    def mixed_image(self):
        if self.mode == 'mp' or self.mode == 'ri':
            self._mixed_image = self._mix_images(self.mode)
        else:
            raise ValueError("Invalid mode")

        return self._mixed_image

    @mixed_image.setter
    def mixed_image(self, value):
        self._mixed_image = value

    def _mix_images(self, mode):
        # we use logging to detect (axis=0)
        mixer_logger.debug("Enter mix_images function")
        weighted_phase = []
        weighted_magnitude = []
        weighted_real_part = []
        weighted_imaginary_part = []
        for img in self.input_images:
            weight = img.weight
            img_component = img.current_clipped_image if img.is_captured else img.current_image
            img_state = img.current_state
            weighted_comp = weight * img_component
            if img_state == 'm' and mode == 'mp':
                weighted_magnitude.append(weighted_comp)
            elif img_state == 'p' and mode == 'mp':
                weighted_phase.append(weighted_comp)
            elif img_state == 'r' and mode == 'ri':
                weighted_real_part.append(weighted_comp)
            elif img_state == 'i' and mode == 'ri':
                weighted_imaginary_part.append(weighted_comp)
            else:
                continue
        weighted_fft = 0
        if mode == 'mp':
            mixer_logger.debug("Enter if(mode == 'mp') at mix_images function")
            weighted_magnitude = np.sum(np.array(weighted_magnitude), axis=0)
            weighted_phase = np.sum(np.array(weighted_phase), axis=0)
            weighted_fft = weighted_magnitude * np.exp(1j * weighted_phase)

        else:
            weighted_real_part = np.sum(np.array(weighted_real_part), axis=0)
            weighted_imaginary_part = np.sum(np.array(weighted_imaginary_part), axis=0)
            weighted_fft = weighted_real_part + weighted_imaginary_part * 1j
        self.mixed_fft = weighted_fft
        _mixed_image = np.abs(np.fft.ifft2(self.mixed_fft)).astype(np.uint8)
        mixer_logger.debug("calculate inverse fft for mix_images function")
        _mixed_image = cv2.resize(_mixed_image, (500, 270), cv2.INTER_LINEAR)

        return _mixed_image
