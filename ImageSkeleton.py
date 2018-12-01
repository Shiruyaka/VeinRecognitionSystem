import cv2 as cv
import numpy as np
from collections import Counter
import time

class ImageSkeleton:

    def __init__(self, image):
        self.image = image
        self.mean_nimage = 0
        self._normalized_image = image
        self._directional_map = np.zeros(self.image.shape)
        self._enhanced_image = None
        print("tiuuuuu", self._directional_map.shape)

    def normalize_image(self):
        print("Start normalize")
        mean_0 = 150
        var_0 = 255

        norm_coeff = self.image.shape[0] * self.image.shape[1]
        mean = np.sum(self.image)/norm_coeff

        sq_diff_mean = np.square(self.image - mean)

        var = np.sum(sq_diff_mean)/norm_coeff

        good_var_img = np.sqrt(var_0/var * sq_diff_mean)
        higher_mean_img = mean_0 + (self.image >= mean) * good_var_img
        lower_mean_img = mean_0 - (self.image < mean) * good_var_img

        self._normalized_image = (lower_mean_img + higher_mean_img).astype(np.uint8)
        self.mean_nimage = np.sum(self._normalized_image, axis=(0, 1))/norm_coeff
        print("End normalize")
    def find_direction(self, mean_direction):
        delta_mean = [np.abs(mean_direction[i] - mean_direction[i + 4]) for i in range(4)]
        direction = np.argmax(delta_mean)

        return direction, direction + 4

    # change it into masks and multiply only
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
        values, counts = np.unique(window, return_counts=True)
        return values[np.argmax(counts)]

    def make_directional_image(self):
        PADDING = 4
        padded_image = np.pad(self._normalized_image, (4, 4), 'constant', constant_values=(0, 0))

        for i in range(PADDING, self._directional_map.shape[0]):
            start_time = time.time()
            for j in range(PADDING, self._directional_map.shape[1]):

                mean_direction = self.compute_mean_direction(padded_image, i, j)
                j1, j2 = self.find_direction(mean_direction)
                if np.abs(self.mean_nimage - mean_direction[j1]) < np.abs(self.mean_nimage - mean_direction[j2]):
                    self._directional_map[i - PADDING, j - PADDING] = j1 + 1
                else:
                    self._directional_map[i - PADDING, j - PADDING] = j2 + 1
            print("--- %s seconds ---" % (time.time() - start_time))
        self._directional_map = self._directional_map.astype(np.uint8)

    def smooth_image(self):
        PADDING = 4
        padded_image = np.pad(self._directional_map, (PADDING, PADDING), 'constant', constant_values=(0, 0))
        smoothed_image = np.zeros(self._directional_map.shape)

        for i in range(PADDING, smoothed_image.shape[0]):
            for j in range(PADDING, smoothed_image.shape[1]):
                smoothed_image[i, j] = self.find_locally_direction(padded_image[i - 4: i + 5, j - 4: j + 5])

        self._directional_map = smoothed_image.astype(np.uint8)
        # cv.imshow("smoothed directional map", self._directional_map * 10)
    def skeletonize(self):

        args = {'a': 9, 'b': 0, 'c': 24, 'd': 30}
        orf = OrientedFilter(args)

        self.normalize_image()
        self.make_directional_image()
        self.smooth_image()
        print(self._directional_map.shape)
        self._enhanced_image = orf.filter(self._normalized_image, self._directional_map)
        cv.imshow("enhanced image", self._enhanced_image)
        cv.waitKey(0)

# cv.imshow("original_image", test)
# imgsk.normalize_image()
# # cv.imshow("normalized_image", imgsk._normalized_image)
# imgsk.make_directional_image(imgsk._normalized_image)
# imgsk.smooth_image()
# cv.waitKey(0)
#
# x = np.array([[1,2,3],[5,6,7],[8,9,1]])
# print(x[0:2, 1:2])
#
# x = [1,2,3,4,5,6]
# y = [1,2,3,4,5,6]
#
# print(list(zip(x, y)))

class OrientedFilter:

    def __init__(self, coeff):
        self.base_mask = self.make_oriented_filter(coeff)
        self.directional_masks = {i : self.rotate_mask( np.pi/i) for i in range(1, 8)}
        self.directional_masks[0] = self.base_mask[2:9, 2:9]

    def make_oriented_filter(self, coeff):

        assert(coeff['d'] + 2*coeff['a'] + 2*coeff['b'] - 2*coeff['c'] == 0)
        assert(coeff['c'] > 0 and coeff['d'] >= 0 and coeff['d'] > coeff['a'])

        a, b, c, d = coeff['a'], coeff['b'], coeff['c'], coeff['d']
        return np.array(
            [
                [c/3, 0, -c/3, -2*c/3, -c, -c, -c, -2*c/3, -c/3, 0, c/3],
                [c/3, 0, -c/3, -2*c/3, -c, -c, -c, -2*c/3, -c/3, 0, c/3],
                [c/3, 0, -c/3, -2*c/3, -c, -c, -c, -2*c/3, -c/3, 0, c/3],
                [-b/3, 0, b/3, 2*b/3, b, b, b, 2*b/3, b/3, 0, -b/3],
                [-a/3, 0, a/3, 2*a/3, a, a, a, 2*a/3, a/3, 0, -a/3],
                [-d/3, 0, d/3, 2*d/3, d, d, d, 2*d/3, d/3, 0, -d/3],
                [-a/3, 0, a/3, 2*a/3, a, a, a, 2*a/3, a/3, 0, -a/3],
                [-b/3, 0, b/3, 2*b/3, b, b, b, 2*b/3, b/3, 0, -b/3],
                [c/3, 0, -c/3, -2*c/3, -c, -c, -c, -2*c/3, -c/3, 0, c/3],
                [c/3, 0, -c/3, -2*c/3, -c, -c, -c, -2*c/3, -c/3, 0, c/3],
                [c/3, 0, -c/3, -2*c/3, -c, -c, -c, -2*c/3, -c/3, 0, c/3]
            ]
        )

    def interpolate(self, coords):
        #make square
        movement = np.array([5, 5])
        ru = np.ceil(coords).astype(int)
        ld = np.floor(coords).astype(int)
        rd = np.array([ru[0], ld[1]], dtype=int)
        lu = np.array([ld[0], ru[1]], dtype=int)

        dx = (coords[0] - lu[0])
        dy = (coords[1] - ld[1])

        if tuple(ru) == tuple(coords):
            return self.base_mask.item(tuple(ru))
        else:
            ru_id = tuple(ru + movement)
            ld_id = tuple(ld + movement)
            rd_id = tuple(rd + movement)
            lu_id = tuple(lu + movement)

            up_inter = self.base_mask.item(lu_id) +  dx * (self.base_mask.item(ru_id) - self.base_mask.item(lu_id))
            down_inter = self.base_mask.item(ld_id) + dx * (self.base_mask.item(rd_id) - self.base_mask.item(ld_id))

            return down_inter + dy * (up_inter - down_inter)
    def rotate_mask(self, angle):
        rotated_mask = np.zeros((7,7))
        rotation_matrix = [[np.cos(angle), np.sin(angle)],
                          [-np.sin(angle), np.cos(angle)]]

        for i in range(-3, 3, 1):
            for j in range(-3, 3, 1):
                base_mask_coord = np.dot(rotation_matrix, np.array([i, j]))
                # print(base_mask_coord)
                rotated_mask[i + 3, j + 3] = self.interpolate(base_mask_coord)
        
        return rotated_mask
    
    def filter(self, image, directional_map):
        PADDING = 3
        padded_image = np.pad(image, (PADDING, PADDING), 'constant', constant_values=(0, 0))
        output = np.zeros(image.shape)

        for i in range(3, image.shape[0], 1):
            for j in range(3, image.shape[1], 1):

                mask = self.directional_masks[directional_map[i, j]]
                output[i - 3, j - 3] =  np.sum(mask * padded_image[i - 3: i + 4, j - 3: j + 4]) 

        return output


test = cv.imread("rabbit.bmp", 0)
print(test.shape)
imgsk = ImageSkeleton(test)
imgsk.skeletonize()


