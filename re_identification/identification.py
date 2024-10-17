import sys

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

sys.path.append('methods/SOLIDER_REID')
sys.path.append('methods/FaceDetection_DSFD')
from model import make_model
from datasets.make_dataloader import make_custom_dataloader
from datasets.transformation import CustomTransform
from processor import do_custom_inference
from face_ssd_infer import SSD


class ReIdentificationManager:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.face_detection_model = None
        self.solider_model = None
        self.use_cv2 = False

    def load_models(self) -> None:
        self.load_solider_model()
        if self.cfg.TEST.USE_FACE_DETECTION:
            self.load_face_detection_model()

    def load_solider_model(self) -> None:
        self.solider_model = make_model(self.cfg, num_class=0, camera_num=0, view_num=0,
                                        semantic_weight=self.cfg.MODEL.SEMANTIC_WEIGHT)

        self.solider_model.load_param(self.cfg.TEST.WEIGHT)

    def load_face_detection_model(self) -> None:
        self.face_detection_model = SSD("test")
        path = 'methods/FaceDetection_DSFD/pretrained_models/WIDERFace_DSFD_RES152.pth'
        self.face_detection_model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))

    def get_face_detection_model(self) -> nn.Module:
        return self.face_detection_model

    def get_solider_model(self) -> nn.Module:
        return self.solider_model

    def load_data(self, val_transforms=None, query_from_gui=None):

        val_loader, num_query, cam_num, track_num = make_custom_dataloader(self.cfg,
                                                                           val_transforms=val_transforms,
                                                                           query_from_gui=query_from_gui,
                                                                           use_cv2=self.use_cv2)

        return val_loader, num_query, cam_num, track_num

    def do_inference(self, query_from_gui: Image) -> None:
        if self.cfg.TEST.USE_FACE_DETECTION:
            self.use_cv2 = True
            custom_t = CustomTransform(self.cfg.INPUT.SIZE_TEST, self.face_detection_model.test_transform)
            val_transforms = T.Compose([custom_t])
        else:
            self.face_detection_model = None
            self.use_cv2 = False
            self.cfg.defrost()
            self.cfg.TEST.IMS_PER_BATCH = 32
            self.cfg.freeze()
            val_transforms = T.Compose([
                T.Resize(self.cfg.INPUT.SIZE_TEST),
                T.ToTensor(),
                T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)
            ])

        val_loader, num_query, _, _ = self.load_data(val_transforms=val_transforms,
                                                     query_from_gui=query_from_gui)
        return do_custom_inference(self.cfg,
                                   self.solider_model,
                                   self.face_detection_model,
                                   val_loader,
                                   num_query,
                                   query_from_gui)
