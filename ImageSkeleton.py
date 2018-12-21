import cv2 as cv
import numpy as np
from collections import Counter
import time
from OrientedFilter import OrientedFilter
from Utils import make_mask_directions, trim_image, trim_image, make_size_tuple
from skimage.filters import threshold_niblack
from skimage.morphology import skeletonize
from skimage.util import invert
from numba import njit

class ImageSkeleton:

    def __init__(self, image):
        self.image = image
        self.mean_nimage = 0
        self._normalized_image = image
        self._directional_map = np.zeros(self.image.shape)
        self._enhanced_image = None
        self._masks_direction = make_mask_directions()
    
    @property
    def image_dim(self):
        return self.image.shape

    def normalize_image(self):
        mean_0 = 150
        var_0 = 255

        mean = np.mean(self.image)
        sq_diff_mean = (self.image - mean)**2
        var = np.var(self.image)

        good_var_img = np.sqrt(var_0/var * sq_diff_mean)
        higher_mean_img = mean_0 * (self.image >= mean) + (self.image >= mean) * good_var_img
        lower_mean_img = mean_0 * (self.image < mean) - (self.image < mean) * good_var_img

        self._normalized_image = (lower_mean_img + higher_mean_img)
        self.mean_nimage = mean


    def direction_convolution(self, padded_img):
        # anchor = (3 , 3)
        return np.array([cv.filter2D(padded_img, -1, mask, borderType=cv.BORDER_ISOLATED)/4 
                            for mask in self._masks_direction])

    def mul_one(self, no, padded_image):

        mask = self._masks_direction[no]
        output = np.zeros(self.image.shape)

        for i in range(4, self.image.shape[0]):
            for j in range(4, self.image.shape[1]):
                output[i - 4, j - 4] = np.sum(padded_image[i - 4 : i + 5, j - 4 : j + 5] * mask)/4
        
        return output

    def compute_mean_direction(self):
        PADDING = 4
        shape = make_size_tuple(self._normalized_image.shape, PADDING)

        padded_image = np.pad(self._normalized_image, (PADDING, PADDING), 'constant', constant_values=(120.69533435, 120.69533435))
        padded_image = padded_image.astype(np.int64)
        means_direction = self.direction_convolution(padded_image)
        means_direction = np.array([trim_image(img, shape) for img in means_direction])

        return means_direction

    def find_vein_direction(self, means_direction):
        d_orth_difference = np.array([np.abs(means_direction[i, :, :] - means_direction[i + 4, :, :]) for i in range(4)])
        print("vein direction")
        mean = self.mean_nimage
        print(d_orth_difference[:,0,0])
        max_delta = np.argmax(d_orth_difference, axis=0)
        print(max_delta[0,0])
        pixel_direction = []
        for i in range(4):
            mask = np.abs(mean - means_direction[i, :, : ]) < np.abs(mean - means_direction[i + 4, :, : ] )
            vector_map = mask * (i + 1)
            vector_map[vector_map == 0] = i + 5

            pixel_direction.append(vector_map)
        
        output = np.zeros(self.image.shape)
        
        idx = 0
        for directions in pixel_direction:
            output += (max_delta == idx) * directions
            idx += 1
        
        return output
    
    def smoothing_directional_map(self, direction_map, window_size):
        i_max, j_max = direction_map.shape
        conitnous_directional_map = np.zeros(direction_map.shape)
        
        for i in range(0, i_max, 1):
            for j in range(0, j_max, 1):

                i_s = max(int(i-window_size/2), 0)
                i_e = min(int(i+window_size/2), i_max)
                j_s = max(int(j-window_size/2), 0)
                j_e = min(int(j+window_size/2), j_max)

                idx, counts = np.unique(direction_map[i_s : i_e, j_s : j_e], return_counts=True)
                conitnous_directional_map[i, j] = idx[np.argmax(counts)]
                
        
        return conitnous_directional_map

    def make_directional_image(self):
        
        means_directions = self.compute_mean_direction()
        print("make directional image fuunc " + str(type(means_directions[0,0,0])))
        direction_map = self.find_vein_direction(means_directions)
        self._directional_image = self.smoothing_directional_map(direction_map, 8).astype(np.uint8)
        
    def round_image(self, image):
        min_pixi = np.amin(image)
        max_pixi = np.amax(image)

        normalization_coeff = max_pixi - min_pixi
        output = (image - min_pixi)/normalization_coeff * 255
        return output.astype(np.uint8)

    def make_coeffs(self, a, b, d):
        return {'a': a, 'b': b, 'c': (d + 2*a + 2*b)/2, 'd': d}
    def skeletonize(self):

        args = {'a': 9, 'b': 4, 'c': 24, 'd': 30}
        args = self.make_coeffs(9, 0, 30)
        orf = OrientedFilter(args)

        self.normalize_image()
        print(self._normalized_image[:10, :10])
        self.make_directional_image()
        cv.imshow("enhanced_img12", self._directional_image * 10)
        enhanced_img = orf.filter(self.image, self._directional_image)
        print(np.unique(enhanced_img, return_counts=True))
        self.enhanced_img_ = self.round_image(enhanced_img)
        print(np.unique(self.enhanced_img_, return_counts=True))
        cv.imshow("enhanced_img", self.enhanced_img_)
        thresh_niblack = threshold_niblack(self.enhanced_img_, window_size=9, k=0.01)
        segm_img = self.enhanced_img_ < thresh_niblack
        self._enhanced_image = segm_img.astype(np.uint8) * 255
        cv.imshow("enhanced image", self._enhanced_image)
        


test = cv.imread("rabbit.bmp", 0)
# test = cv.resize(test, None, fx=0.5, fy=0.5)
# cv.imshow("normal img", test)
print(test[:10, :10])
imgsk = ImageSkeleton(np.array(test))
imgsk.skeletonize()

cv.waitKey(0)

