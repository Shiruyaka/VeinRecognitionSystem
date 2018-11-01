import cv2 as cv
import numpy as np
from collections import Counter

class ImageSkeleton:

    def __init__(self, image):
        self.image = image
        self.mean_nimage = 0
        self._normalized_image = None
        self._directional_map = np.zeros(self.image.shape)

    def normalize_image(self):
        mean_0 = 150
        var_0 = 255

        norm_coeff = self.image.shape[0] * self.image.shape[1]
        mean = np.sum(self.image, axis=(0, 1))/norm_coeff

        sq_diff_mean = np.square(self.image - mean)

        var = np.sum(sq_diff_mean, axis=(0, 1))/norm_coeff

        good_var_img = np.sqrt(var_0/var * sq_diff_mean)
        higher_mean_img = mean_0 + (self.image >= mean) * good_var_img
        lower_mean_img = mean_0 - (self.image < mean) * good_var_img

        self._normalized_image = (lower_mean_img + higher_mean_img).astype(np.uint8)
        self.mean_nimage = np.sum(self._normalized_image, axis=(0, 1))/norm_coeff

    def find_direction(self, mean_direction):

        delta_mean = [np.abs(mean_direction[i] - mean_direction[i + 4]) for i in range(4)]
        direction = np.argmax(delta_mean)

        return direction, direction + 4

    def compute_mean_direction(self, padded_image, i, j):

        mean_direction = np.zeros(8)

        for k in range(5):

            mean_direction[0] += padded_image[i, j - 4 + k * 2]
            mean_direction[1] += padded_image[i + 2 - k, j - 4 + k * 2]
            mean_direction[2] += padded_image[i + 4 - 2 * k, j - 4 + 2 * k]
            mean_direction[3] += padded_image[i + 4 - 2 * k, j - 2 + k]
            mean_direction[4] += padded_image[i - 4 + k * 2, j]
            mean_direction[5] += padded_image[i - 4 + 2 * k, j - 2 + k]
            mean_direction[6] += padded_image[i - 4 + 2 * k, j - 4 + 2 * k]
            mean_direction[7] += padded_image[i - 2 + k, j - 4 + 2 * k]

        mean_direction -= padded_image[i, j]
        mean_direction /= 4

        return mean_direction

    def find_locally_direction(self, window):
        histogram = sorted(zip(np.unique(window, return_counts=True)), key=lambda x : x[1], reverse=True)
        if histogram[0][0] != 0:
            return histogram[0][0]
        else:
            return histogram[1][0]

    def make_directional_image(self, normalized_image):
        PADDING = 4
        padded_image = np.pad(normalized_image, (4, 4), 'constant', constant_values=(0, 0))

        for i in range(PADDING, padded_image.shape[0] - PADDING):
            for j in range(PADDING, padded_image.shape[1] - PADDING):

                mean_direction = self.compute_mean_direction(padded_image, i, j)
                j1, j2 = self.find_direction(mean_direction)

                if np.abs(self.mean_nimage - mean_direction[j1]) < np.abs(self.mean_nimage - mean_direction[j2]):
                    self._directional_map[i, j] = j1 + 1
                else:
                    self._directional_map[i, j] = j2 + 5

        cv.imshow("directional_map", self._directional_map)
        cv.waitKey(0)

    def smooth_image(self):
        # padded_image =
        # for
        pass
# test = cv.imread("rabbit.bmp", 0)
# imgsk = ImageSkeleton(test)
# cv.imshow("original_image", test)
# imgsk.normalize_image()
# cv.imshow("normalized_image", imgsk._normalized_image)
# cv.imshow("directional_map", imgsk._directional_map)
# cv.waitKey(0)
# unique, counts = np.unique(np.array(np.arange(9).reshape((3, 3))), return_counts=True)
# print(unique, counts)