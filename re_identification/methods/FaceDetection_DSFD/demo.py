import cv2
import torch

from face_ssd_infer import SSD
from utils import vis_detections

device = torch.device("cuda")
conf_thresh = 0.5

net = SSD("test")
net.load_state_dict(torch.load('pretrained_models/WIDERFace_DSFD_RES152.pth'))
net.to(device).eval()

img_path = 'imgs/SAT_Camera1_MILLIS_561227_TRK-ID_204_TIMESTAMP_0-9-21.png'

img = cv2.imread(img_path, cv2.IMREAD_COLOR)
h, w, _ = img.shape
detections = net.detect_on_image([img, img], (h, w), device, is_pad=False, keep_thresh=conf_thresh)
print(detections.shape)
vis_detections(img, detections[0], conf_thresh, show_text=False)
vis_detections(img, detections[1], conf_thresh, show_text=False)
