import os
import cv2 as cv

def export_to_png(name, data):
    folder = "gallery"
    if not os.path.exists(folder):
        os.mkdir(folder)
    file = os.path.join(folder, f"{name}.png")
    cv.imwrite(file, data)