import numpy as np
import cv2


class CustomTransform(object):
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size
        self.interpolation = 3

    def __call__(self, image):
        image = cv2.resize(image,
                           (self.target_size[1], self.target_size[0]),
                           interpolation=self.interpolation)

        return image
