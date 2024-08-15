import numpy as np
import cv2


class CustomTransform(object):
    def __init__(self, target_size, scale_from_original=True):
        super().__init__()
        self.scales = []
        self.target_size = target_size
        self.interpolation = 3
        self.scale_from_original = scale_from_original

    def __call__(self, image):
        image = np.array(image)[:, :, (2, 1, 0)]

        curr_img_size = image.shape[:2]
        image = cv2.resize(image,
                           (self.target_size[1], self.target_size[0]),
                           interpolation=self.interpolation)

        resize_factor_y = self.target_size[1] / curr_img_size[1]
        resize_factor_x = self.target_size[0] / curr_img_size[0]

        if self.scale_from_original:
            self.scales.append(np.array([self.target_size[1] / resize_factor_y,
                                         self.target_size[0] / resize_factor_x,
                                         self.target_size[1] / resize_factor_y,
                                         self.target_size[0] / resize_factor_x]))
        else:
            self.scales.append(np.array([self.target_size[1],
                                         self.target_size[0],
                                         self.target_size[1],
                                         self.target_size[0]]))

        return image

    def get_scales(self):
        return np.array(self.scales)
