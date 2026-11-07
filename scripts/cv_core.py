"""Classical registration primitives; no model downloads or image generation."""

import time
import cv2 as cv
import numpy as np
from PIL import Image

T0 = time.monotonic()


def log(message):
    print(f'[{time.monotonic() - T0:6.1f}s] {message}', flush=True)


def save_rgb(path, array):
    Image.fromarray(array).save(path)


def blur(a, sigma):
    return cv.GaussianBlur(a, (0, 0), sigma, borderType=cv.BORDER_REFLECT_101)


def smoothstep(value, lo, hi):
    x = np.clip((value - lo) / (hi - lo), 0, 1)
    return x * x * (3 - 2 * x)


def normalize_for_flow(rgb):
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    gray = cv.createCLAHE(clipLimit=1.4, tileGridSize=(12, 18)).apply(gray)
    return cv.GaussianBlur(gray, (0, 0), 0.65)


def dense_flow(first, second):
    engine = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
    engine.setFinestScale(0)
    engine.setPatchSize(12)
    engine.setPatchStride(4)
    engine.setGradientDescentIterations(40)
    engine.setVariationalRefinementIterations(12)
    engine.setUseMeanNormalization(True)
    result = engine.calc(first, second, None)
    return blur(result, 1.15)


def correlation(first, second, sigma=9):
    first = blur(first, 0.85) - blur(first, 12)
    second = blur(second, 0.85) - blur(second, 12)
    ma, mb = blur(first, sigma), blur(second, sigma)
    aa = np.maximum(blur(first * first, sigma) - ma * ma, 0)
    bb = np.maximum(blur(second * second, sigma) - mb * mb, 0)
    ab = blur(first * second, sigma) - ma * mb
    return np.clip(ab / np.sqrt(aa * bb + 0.3), -1, 1)
