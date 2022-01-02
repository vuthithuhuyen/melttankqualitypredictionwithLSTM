import os

import cv2
import numpy
import numpy as np
from PyQt5.QtWidgets import QSlider, QLabel, QComboBox


from Helper.OpenCVHelper import ReadImage, Detect_Edge, rescale_frame
from Helper.PyQTHelper import QLabelDisplayImage
from Model import GlobalVariables
from Model.GlobalVariables import display_rescale




# slider train/test split change value
def sliderChangeValue(slider: QSlider, label: QLabel, convert_percent=False):
    try:
        val = slider.value()
        if convert_percent:
            val /= 100
        label.setText(str(val))
    except Exception as e:
        print(e)


# Initial combox state
def InitialComboxState(cmb: QComboBox, _values):
    try:
        states = _values
        cmb.clear()
        cmb.addItems(states)
    except Exception as e:
        print(e)
