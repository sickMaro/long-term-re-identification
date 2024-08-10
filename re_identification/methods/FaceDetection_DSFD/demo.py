import cv2
import torch

from face_ssd_infer import SSD
from utils import vis_detections

device = torch.device("cuda")
conf_thresh = 0.5

net = SSD("test")
net.load_state_dict(torch.load('pretrained_models/WIDERFace_DSFD_RES152.pth'))
net.to(device).eval()

img_path = 'imgs/SAT_Camera1_MILLIS_3003_TRK-ID_2_TIMESTAMP_0-0-3.png'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
detections = net.detect_on_image([img], (374, 128), device, keep_thresh=conf_thresh)

vis_detections(cv2.resize(img, (128, 374)), detections[0], conf_thresh, show_text=False)
