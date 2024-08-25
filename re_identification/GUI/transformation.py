import cv2
import torch


class CustomTransform(object):
    def __init__(self, target_size, transform):
        super().__init__()
        self.target_size = target_size
        self.interpolation = 3
        self.transform = transform

    def __call__(self, image):
        image = cv2.resize(image,
                           (self.target_size[1], self.target_size[0]),
                           interpolation=self.interpolation)
        image = self.transform(image)
        image = torch.from_numpy(image).permute(2, 0, 1)

        return image
