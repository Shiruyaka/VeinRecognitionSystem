import cv2 as cv
import numpy as np


class ImageSkeleton:

    def __init__(self, image):
        self.image = image

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

        return (lower_mean_img + higher_mean_img).astype(np.uint8)


test = cv.imread("rabbit.bmp", 0)
imgsk = ImageSkeleton(test)
cv.imshow("kurwa świnia", test)
cv.imshow("arka gdynia", imgsk.normalize_image())
cv.waitKey(0)
