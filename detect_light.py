import numpy as np
from scipy import signal
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from PIL import Image

RED = 0
GREEN = 1

N = 150


def init_red_kernel():
    return np.array([[-1 / 9, -1 / 9, -1 / 9],
                     [-1 / 9, 8 / 9, -1 / 9],
                     [-1 / 9, -1 / 9, -1 / 9]])


def init_green_kernel():
    return np.array([[-1 / 9, -1 / 9, -1 / 9],
                     [-1 / 9, 8 / 9, -1 / 9],
                     [-1 / 9, -1 / 9, -1 / 9]])


def print_image(image):
    img = Image.fromarray(image, 'RGB')
    img.show()


def filter_by_arg(max_filtered, grad, arg):
    _filter = np.argwhere(grad > arg)
    max_ = np.argwhere(max_filtered == grad)
    _filter = set((tuple(i) for i in _filter))
    max_ = set((tuple(i) for i in max_))

    res = _filter.intersection(max_)

    x = list(map(lambda x: x[0], res))
    y = list(map(lambda x: x[1], res))

    return x, y


def find_n_max_pixels(c_image):
    # norm_image = c_image/255
    # pixel = np.sort(c_image.ravel())
    # len_ = len(pixel)
    # range_ = abs(pixel[-1]) - abs(pixel[0]) if abs(pixel[-1]) > abs(pixel[0]) else abs(pixel[0]) - abs(pixel[-1])
    # distribution = len_ / range_
    # np.argwhere(c_image < pixel[0])
    # np.argwhere(c_image > pixel[-1])
    # return pixel[-N]

    max_pixel = 29

    return max_pixel


def print_convolve(image, grad):
    fig, (ax_orig, ax_mag) = plt.subplots(2, 1, figsize=(6, 15))

    ax_orig.imshow(image, cmap='gray')
    ax_orig.set_axis_off()

    ax_mag.imshow(np.absolute(grad.T), cmap='gray')
    ax_mag.set_axis_off()

    fig.show()


def find_tfl_lights_by_color(tfl, color):
    grad = tfl.convolve_by_color(tfl.image, color)

    g = ndimage.maximum_filter(grad, size=20)

    x, y = filter_by_arg(g, grad, 25)

    return x, y


def find_tfl_lights(c_image: np.ndarray):
    """
    The image itself as np.uint8, shape of (H, W, 3)
    :param c_image:
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """

    tfl = TFL(c_image)

    green_x, green_y = find_tfl_lights_by_color(tfl, GREEN)
    red_x, red_y = find_tfl_lights_by_color(tfl, RED)

    return red_x, red_y, green_x, green_y


class TFL:

    def __init__(self, image=None):
        self.kernel = {GREEN: init_green_kernel(), RED: init_red_kernel()}
        self.image = image

    def convolve_by_color(self, image, color):
        gray_image = image[:, :, color]
        grad = signal.convolve2d(gray_image.T, self.kernel[color], boundary='symm', mode='same')

        return grad

    def height(self):
        return self.image.shape[0]

    def weight(self):
        return self.image.shape[1]
