import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from Utils import make_mask_directions, trim_image, make_size_tuple
from skimage import transform as tf


class OrientedFilter:

    def __init__(self, coeff):
        self.base_mask = self.make_oriented_filter(coeff)
        # self.base_mask = self.make_bigger_array(coeff)
        one_part = 180/8
        self.directional_masks2 = {}
        print(one_part)
        self.directional_masks = \
            {angle + 1 : ndimage.rotate(self.base_mask, angle * one_part, reshape=False, mode="nearest") for angle in range(0, 8, 1)}
        for angle in range(1, 8, 1):
            tform = tf.SimilarityTransform(rotation = angle * np.pi/8)
            rotated = tf.warp(self.base_mask, tform, order=0)
            self.directional_masks2[angle + 1] = rotated

        self.directional_masks2[1] = self.base_mask
    def make_oriented_filter(self, coeff):

        assert(coeff['d'] + 2*coeff['a'] + 2*coeff['b'] - 2*coeff['c'] == 0)
        

        a, b, c, d = coeff['a'], coeff['b'], coeff['c'], coeff['d']
        return np.array(
            [
               
                [-c/3, -2*c/3, -c, -c, -c, -2*c/3, -c/3],
                [b/3, 2*b/3, b, b, b, 2*b/3, b/3],
                [a/3, 2*a/3, a, a, a, 2*a/3, a/3],
                [d/3, 2*d/3, d, d, d, 2*d/3, d/3],
                [a/3, 2*a/3, a, a, a, 2*a/3, a/3],
                [b/3, 2*b/3, b, b, b, 2*b/3, b/3],
                [-c/3, -2*c/3, -c, -c, -c, -2*c/3, -c/3]
               
            ]
        )
    def make_bigger_array(self, coeff):
        a, b, c, d = coeff['a'], coeff['b'], coeff['c'], coeff['d']
        return np.array(
            [
                [c/3, 0, -c/3, -2*c/3, -c, -c, -c, -2*c/3, -c/3, 0, c/3],
                [c/3, 0, -c/3, -2*c/3, -c, -c, -c, -2*c/3, -c/3, 0, c/3],
                [-b/3, 0, b/3, 2*b/3, b, b, b, 2*b/3, b/3, 0, -b/3],
                [-a/3, 0, a/3, 2*a/3, a, a, a, 2*a/3, a/3, 0, -a/3],
                [-d/3, 0, d/3, 2*d/3, d, d, d, 2*d/3, d/3, 0, -d/3],
                [-d/3, 0, d/3, 2*d/3, d, d, d, 2*d/3, d/3, 0, -d/3],
                [-d/3, 0, d/3, 2*d/3, d, d, d, 2*d/3, d/3, 0, -d/3],
                [-a/3, 0, a/3, 2*a/3, a, a, a, 2*a/3, a/3, 0, -a/3],
                [-b/3, 0, b/3, 2*b/3, b, b, b, 2*b/3, b/3, 0, -b/3],
                [c/3, 0, -c/3, -2*c/3, -c, -c, -c, -2*c/3, -c/3, 0, c/3],
                [c/3, 0, -c/3, -2*c/3, -c, -c, -c, -2*c/3, -c/3, 0, c/3]
            ]
        )
    
    def filter(self, image, directional_map):
        PADDING = 3
        padded_image = np.pad(image, (PADDING, PADDING), 'constant', constant_values=(1, 1))

        desired_shape = make_size_tuple(image.shape, PADDING)
        # anchor = (4, 4)
        dir_mask_on_img = \
            {i : trim_image(cv.filter2D(padded_image, -1, ker, borderType = cv.BORDER_ISOLATED), desired_shape)
                for i, ker in self.directional_masks2.items()}

        output = np.zeros(image.shape).astype(np.int64)

        for i in range(1, 9, 1):
            output += dir_mask_on_img[i] * (directional_map == i)

        return output
