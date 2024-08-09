import os
import sys

import torch
import torch.nn as nn
import logging
sys.path.append('../methods/SOLIDER_REID')
sys.path.append('../methods/FaceDetection_DSFD')
from datasets.make_dataloader import make_custom_dataloader
from model import make_model
from processor import do_custom_inference
from utils.metrics import R1_mAP_eval, CustomEvaluator
from face_ssd_infer import SSD


class ReIdentificationManager:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.face_detection_model = None
        self.solider_model = None
        self.conf_threshold = 0.5

    def load_solider_model(self):
        self.solider_model = make_model(self.cfg, num_class=0, camera_num=0, view_num=0,
                                        semantic_weight=self.cfg.MODEL.SEMANTIC_WEIGHT)

        self.solider_model.load_param(self.cfg.TEST.WEIGHT)

    def load_face_detection_model(self):
        self.face_detection_model = SSD("test")
        (self.face_detection_model.load_state_dict
         (torch.load('../methods/FaceDetection_DSFD/pretrained_models/WIDERFace_DSFD_RES152.pth')))

    def get_face_detection_model(self):
        return self.face_detection_model

    def get_solider_model(self):
        return self.solider_model

    def inference_with_solider(self, query_from_gui):
        val_loader, num_query, _, _ = make_custom_dataloader(self.cfg, query_from_gui=query_from_gui)

        return do_custom_inference(self.cfg, self.solider_model, val_loader, num_query)

    def inference_with_face_det_and_solider(self, query_from_gui):
        val_loader, num_query, _, _ = make_custom_dataloader(self.cfg, query_from_gui=query_from_gui)

        device = "cuda"
        logger = logging.getLogger("transreid.test")
        logger.info("Enter inferencing")

        evaluator = CustomEvaluator(num_query, max_rank=50, feat_norm=self.cfg.TEST.FEAT_NORM,
                                    reranking=self.cfg.TEST.RE_RANKING)
        evaluator.reset()

        if device:
            if torch.cuda.device_count() > 1:
                print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
                self.solider_model = nn.DataParallel(self.solider_model)
                self.face_detection_model = nn.DataParallel(self.face_detection_model)
            self.solider_model.to(device)
            self.face_detection_model.to(device)

        self.face_detection_model.eval()
        self.solider_model.eval()

        for n_iter, (img, timestamp, camid, camids, trackid, imgpath) in enumerate(val_loader):
            h, w, _ = img.shape
            detections = self.face_detection_model.detect_on_image(img, (h, w), device,
                                                                   is_pad=False, keep_thresh=self.conf_threshold)
            print(detections.shape)
            with torch.no_grad():
                img = img.to(device)
                feat, _ = self.solider_model(img)
                evaluator.update((feat, timestamp, camid, trackid, imgpath))

        logger.info('Starting evaluation')
        distmat, timestamps, camids, trackids, imgs_paths = evaluator.compute()
        return distmat, timestamps, camids, trackids, imgs_paths