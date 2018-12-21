import numpy as np 

direction_mask = np.array([
    [7, 0, 6, 0, 5, 0, 4, 0, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [8, 0, 7, 6, 5, 4, 3, 0, 2],
    [0, 0, 8, 0, 0, 0, 2, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 1], 
    [0, 0, 2, 0, 0, 0, 8, 0, 0], 
    [2, 0, 3, 4, 5, 6, 7, 0, 8], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [3, 0, 4, 0, 5, 0, 6, 0, 7]
])

direction_mask2 = np.array([
    [7, 0, 6, 0, 5, 0, 4, 0, 3],
    [0, 7, 6, 6, 5, 4, 4, 3, 0], 
    [8, 8, 7, 6, 5, 4, 3, 2, 2],
    [0, 8, 8, 7, 5, 3, 2, 2, 0],
    [1, 1, 1, 1, 0, 1, 1, 1, 1], 
    [0, 2, 2, 3, 5, 7, 8, 8, 0], 
    [2, 2, 3, 4, 5, 6, 7, 8, 8], 
    [0, 3, 4, 4, 5, 6, 6, 7, 0], 
    [3, 0, 4, 0, 5, 0, 6, 0, 7]
])
def make_mask_directions():
    return np.array([direction_mask == i for i in range(1, 9, 1)], dtype=np.uint8)

def make_size_tuple(img_shape, padding):
    return (padding, img_shape[0] + padding, padding, img_shape[1] + padding)

def trim_image(image, sizes):
    return image[sizes[0] : sizes[1], sizes[2] : sizes[3]]


