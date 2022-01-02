import cv2
import numpy as np

from Model import GlobalVariables



def IsImage(filename):
    try:
        extension = filename.split('.')[-1]
        if str.lower(extension) in ['jpg', 'png', 'jpeg', 'gif', 'tiff']:
            return True
    except Exception as e:
        return False


def ReadImage(filename):
    try:
        img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        print(e)



def Detect_Edge(img, t1, t2):
    if GlobalVariables.blurImage:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    img_edges = cv2.Canny(img, t1, t2)
    return img_edges

#
def ResizeImage(img, newsize):
    return cv2.resize(img, newsize, interpolation=cv2.INTER_AREA)


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    # resize = cv2.resize(image, (new_w, new_h))
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def RawImageToArray(imageFile, thres1, thres2):
    try:
        img = ReadImage(imageFile)
        img_resized = ResizeImage(img, training_size)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        img_edges = Detect_Edge(gray, thres1, thres2)

        return img_edges.flatten()
    except Exception as e:
        print(e)