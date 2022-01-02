import cv2
import qimage2ndarray
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QMessageBox
from PyQt5 import QtCore


def QLabelDisplayImage(imgLabel: QLabel, img):
    # qformat = QImage.Format_Indexed8
    # if len(img.shape) == 3:
    #     if img.shape[2] == 4:
    #         qformat = QImage.Format_RGBA888
    #     else:
    #         qformat = QImage.Format_RGB888
    # img = QImage(img, img.shape[1], img.shape[0], qformat)
    # img = img.rgbSwapped()
    # imgLabel.setPixmap(QPixmap.fromImage(img))
    # imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = qimage2ndarray.array2qimage(img)
    qpixmap = QPixmap.fromImage(image)
    imgLabel.setPixmap(qpixmap)
    imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    return


def showDialog(title, content):
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Information)
    msgBox.setText(content)
    msgBox.setWindowTitle(title)
    msgBox.setStandardButtons(QMessageBox.Ok)
    returnValue = msgBox.exec()
    # if returnValue == QMessageBox.Ok:
    #     print('OK clicked')


def showErrDialog(title, content):
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Critical)
    msgBox.setText(content)
    msgBox.setWindowTitle(title)
    msgBox.setStandardButtons(QMessageBox.Ok)
    returnValue = msgBox.exec()

