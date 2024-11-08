
import cv2
import torch
from PIL import Image

from face_ssd_infer import SSD
from utils import vis_detections, resize_image

device = torch.device("cuda")
conf_thresh = 0.7

net = SSD("test")
net.load_state_dict(torch.load('pretrained_models/WIDERFace_DSFD_RES152.pth'))
net.to(device).eval()
# import numpy as np
img_path = 'imgs/SAT_Camera1_MILLIS_3003_TRK-ID_2_TIMESTAMP_0-0-3.png'
# img = np.array(Image.open(img_path).convert('RGB'))[:, :, (2, 1, 0)]
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

images, scales = resize_image([img], (350, 120))
images = net.test_transform(images)
images = torch.from_numpy(images).permute(0, 3, 1, 2)
# scales = [(350, 120, 350, 120)]
detections = net.detect_on_images(images, scales, device, keep_thresh=conf_thresh)

vis_detections(img, detections[0], conf_thresh, show_text=False)
