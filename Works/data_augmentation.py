import random

import cv2
import numpy as np
from PIL import ImageFilter


####################
# random blur
####################
def random_blur(image, blur_radius):
    return image.filter(ImageFilter.GaussianBlur(radius=random.random() * blur_radius))


####################
# random clahe
####################
def apply_clahe(image, clip_limit):
    if len(image.shape) == 2:
        shape = image.shape
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        image = clahe.apply(image)
    elif len(image.shape) == 3:
        shape = image.shape
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        image = bgr
    return image.reshape(shape)


def random_clahe(image, min, max):
    do_apply = random.randint(0, 1)
    if do_apply == 1:
        clip_limit = min + (max - min) * random.random()
        image = apply_clahe(image.astype("uint8"), clip_limit=clip_limit)
    return image


####################
# random gamma
####################
def adjust_gamma(image, gamma=1.0):
    shape = image.shape
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table).reshape(shape)


def random_gamma(image, min, max):
    gamma = random.uniform(min, max)
    return adjust_gamma(image.astype("uint8"), gamma=gamma)


####################
# random sharpen
####################
def sharpen(image, ratio=0.5):
    shape = image.shape
    # image = image.reshape((shape[0], shape[1]))
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    unsharp_image = cv2.filter2D(image, -1, kernel)
    unsharp_image = image * (1.0 - ratio) + unsharp_image * ratio
    unsharp_image = unsharp_image.reshape(shape)
    return unsharp_image


def random_sharpen(image, max_ratio=0.5):
    do_sharp = random.randint(0, 1)
    if do_sharp > 0:
        ratio = random.uniform(0.001, max_ratio)
        image = sharpen(image, ratio=ratio)
        image[image > 255] = 255.0
        image[image < 0] = 0.0
    return image



def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def flipv(imgg):
    img2= np.zeros([480, 640, 3], np.uint8)
    for i in range(480):

        img2[i,:]=imgg[480-i-1,:]

    return img2

