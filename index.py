import logging

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
from PyQt5.uic import loadUiType
import cv2
import matplotlib.pyplot as plt
from image_mixer import *

index_logger = logging.getLogger("index.py")

ui, _ = loadUiType('main.ui')


def convert_cv_to_qt(cv_image):
    height, width = cv_image.shape
    bytes_per_line = width
    qt_image = QImage(cv_image.data.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qt_image)


class MainApp(QWidget, ui):
    _show_hide_flag = True

    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.resize(1500, 900)

        self.image_1 = ViewOriginal()
        self.image_2 = ViewOriginal()
        self.image_3 = ViewOriginal()
        self.image_4 = ViewOriginal()

        self.image_weight_1 = ViewWeight()
        self.image_weight_2 = ViewWeight()
        self.image_weight_3 = ViewWeight()
        self.image_weight_4 = ViewWeight()


        self.output_image = None

        self.images = [
            [self.image_1,
             self.image_2,
             self.image_3,
             self.image_4],

            [self.image_weight_1,
             self.image_weight_2,
             self.image_weight_3,
             self.image_weight_4],
        ]

        self.graphics_views = [
            [self.graphicsView_original_1,
             self.graphicsView_original_2,
             self.graphicsView_original_3,
             self.graphicsView_original_4
             ],
            [self.graphicsView_weight_1,
             self.graphicsView_weight_2,
             self.graphicsView_weight_3,
             self.graphicsView_weight_4,
             ],
        ]

        self.combo_boxes = [
            self.comboBox_1,
            self.comboBox_2,
            self.comboBox_3,
            self.comboBox_4,
        ]

        self.weight_sliders = [
            self.weight1_slider,
            self.weight2_slider,
            self.weight3_slider,
            self.weight4_slider,
        ]
        self.outer_region_checkBoxs =[
            self.checkBox_1,
            self.checkBox_2,
            self.checkBox_3,
            self.checkBox_4
        ]

        self.mix_graphics_views = [
            self.graphicsView_mix_2,
            self.graphicsView_mix_1
        ]
        self.progress_bars = [
            self.progressBar_mix_2,
            self.progressBar_mix_1
        ]

        for i, graphics_view_list in enumerate(self.graphics_views):
            for j, graphics_view in enumerate(graphics_view_list):
                graphics_view_layout = QHBoxLayout(graphics_view)
                graphics_view_layout.addWidget(self.images[i][j])
                graphics_view.setLayout(graphics_view_layout)

        self.states = {'m': 'Magnitude',
                       'p': 'Phase',
                       'r': 'Real Part',
                       'i': 'Imaginary Part'}

        self.mix_mode = "mp"
        self.mix_graphics_view_index = 0




        for combo_box in self.combo_boxes:
            combo_box.currentTextChanged.connect(self.state_changed)

        for slider in self.weight_sliders:
            slider.valueChanged.connect(self.change_weight)

        for checkBox in self.outer_region_checkBoxs:
            checkBox.toggled.connect(self.toggle_region)

        self.mix_btn.clicked.connect(self.mix)
        self.m_p_radioButton.toggled.connect(self.toggle_mode)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateProgress)

        self.progressValue = 0

    def startProgress(self):
        duration = 0.08
        steps = 100
        interval = int((duration * 1000) / steps)

        self.progressValue = 0
        self.progress_bars[self.mix_graphics_view_index].setValue(0)

        self.timer.start(interval)

    def updateProgress(self):
        self.progressValue += 1
        self.progress_bars[self.mix_graphics_view_index].setValue(self.progressValue)

        if self.progressValue >= 100:
            self.timer.stop()
            self.mix_btn.setText("Mix")
    def state_changed(self):
        if self.sender().currentText() != "Choose":
            i = self.combo_boxes.index(self.sender())
            if self.images[0][i].image_viewer is None:
                QMessageBox.critical(None, "Error", "Double click to open the image first", QMessageBox.Ok)
                self.sender().setCurrentIndex(0)

            else:
                state = ''
                for k in self.states:
                    if self.states[k] == self.sender().currentText():
                        state = k
                self.images[1][i].image_viewer = self.images[0][i].image_viewer
                self.images[1][i].current_state = state
        else:
            index_logger.warning("User choose (Choose)")

    def change_weight(self):
        i = self.weight_sliders.index(self.sender())
        value = self.sender().value()
        self.images[1][i].weight = value / 100


    def mix(self):
        images_to_mix = []
        for i in range(4):
            if self.images[1][i].current_image is not None:
                images_to_mix.append(self.images[1][i])
        self.output_image = ImageMixer(images_to_mix)
        self.output_image.mode = self.mix_mode


        self.mix_btn.setText("Cancel")


        self.mix_graphics_view_index = int(self.activate_radioButton_1.isChecked())

        self.startProgress()

        scene = QGraphicsScene(self.mix_graphics_views[self.mix_graphics_view_index])
        scene_pixmap_item = QGraphicsPixmapItem()
        self.mix_graphics_views[self.mix_graphics_view_index].setScene(scene)
        scene.addItem(scene_pixmap_item)

        qt_image = self.convert_cv_to_qt(self.output_image.mixed_image)
        pixmap = QPixmap(qt_image)
        scene_pixmap_item.setPixmap(pixmap)


    def toggle_region(self):
        i = self.outer_region_checkBoxs.index(self.sender())
        if self.sender().isChecked():
            index_logger.debug("outer region is: ", self.sender().isChecked())
            self.images[1][i].is_outer_region = True
        else:
            self.images[1][i].is_outer_region = False

    def toggle_mode(self):
        if self.m_p_radioButton.isChecked():
            self.mix_mode = "mp"
        else:
            self.mix_mode = "ri"

    def convert_cv_to_qt(self, cv_image):
        height, width = cv_image.shape
        bytes_per_line = width
        qt_image = QImage(cv_image.data.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qt_image)




def main():
    logging.basicConfig(filename="our_app.log",
                        filemode="a",
                        format="(%(asctime)s) | %(name)s | %(levelname)s : '%(message)s' ",
                        datefmt="%d %B %Y, %H:%M",
                        level=logging.DEBUG)
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()



if __name__ == '__main__':
    main()
