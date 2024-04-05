import cv2
import numpy as np

def stereo_matching(numDisparities, blockSize, frame_right, frame_left, f, B):

    # Inicializar el objeto StereoBM o StereotSGBM
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    # Calcular el mapa de disparidad
    disparity_map = stereo.compute(frame_left, frame_right)

    # Asegurarme que no haya divisiones por cero
    disparity_map[disparity_map == 0] = 0.1

    # Calcular el mapa de profundidad
    depth_map = (f * B) / disparity_map

    return depth_map