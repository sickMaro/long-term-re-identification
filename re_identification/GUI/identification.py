import sys

import torch
import torch.nn as nn
import logging

import torchvision.transforms as T
import numpy as np
from PIL import Image

from transformation import CustomTransform

sys.path.append('../methods/SOLIDER_REID')
sys.path.append('../methods/FaceDetection_DSFD')
from model import make_model
from datasets.make_dataloader import make_custom_dataloader
from processor import do_custom_inference
from utils.metrics import R1_mAP_eval, CustomEvaluator
from face_ssd_infer import SSD


class ReIdentificationManager:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.face_detection_model = None
        self.solider_model = None
        self.conf_threshold = 0.8

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

    def load_data(self, val_transforms=None, query_from_gui=None):
        val_loader, num_query, cam_num, track_num = \
            make_custom_dataloader(self.cfg,
                                   val_transforms=val_transforms,
                                   query_from_gui=query_from_gui)

        return val_loader, num_query, cam_num, track_num

    def inference_with_solider(self, query_from_gui):
        val_transforms = T.Compose([
            T.Resize(self.cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)
        ])
        val_loader, num_query, _, _ = self.load_data(val_transforms=val_transforms,
                                                     query_from_gui=query_from_gui)

        return do_custom_inference(self.cfg, self.solider_model, val_loader, num_query)

    def inference_with_face_det_and_solider(self, query_from_gui):
        custom_transform = CustomTransform(self.cfg.INPUT.SIZE_TEST, scale_from_original=False)
        val_transforms = T.Compose([
            custom_transform,
            self.face_detection_model.test_transform,
            T.ToTensor(),
        ])

        val_loader, num_query, _, _ = self.load_data(val_transforms=val_transforms,
                                                     query_from_gui=query_from_gui)

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

        current_scales = (self.cfg.INPUT.SIZE_TEST[1], self.cfg.INPUT.SIZE_TEST[0],
                          self.cfg.INPUT.SIZE_TEST[1], self.cfg.INPUT.SIZE_TEST[0])

        for n_iter, (img, timestamp, camid, trackid, imgpath) in enumerate(val_loader):

            logger.info(f'Batch : {n_iter + 1}')
            # print(f'batch shape {img.shape}')

            detections = self.face_detection_model.detect_on_images(img,
                                                                    current_scales,
                                                                    device,
                                                                    keep_thresh=self.conf_threshold)
            # print(f'det shape {detections.shape}')
            # use img_path to get original image to get the face from if you use original scaling
            indexes = np.where([det.size > 0 for det in detections])[0]
            print(detections.shape)
            print(indexes)
            if len(indexes) > 0:
                detections = detections[indexes]
                img = img[indexes]

                trackid = np.array(trackid)[indexes]
                camid = np.array(camid)[indexes]
                imgpath = np.array(imgpath)[indexes]
                timestamp = np.array(timestamp)[indexes]
                print(trackid)
                faces = []
                detections_per_image = []
                for i in range(detections.shape[0]):
                    if imgpath[i] != '':
                        image = Image.open(imgpath[i], 'r').convert('RGB')
                    else:
                        image = query_from_gui
                    print(f'original image size {image.size}')
                    image = image.resize((self.cfg.INPUT.SIZE_TEST[1], self.cfg.INPUT.SIZE_TEST[0]))
                    # image = np.array(image)
                    print(f'image size after resize {image.size}')
                    for j in range(detections[i].shape[0]):
                        x0, y0, x1, y1 = detections[i][j, :4].astype(int)
                        # face = image[i, :, y0:y1, x0:x1]
                        face = image.crop((x0, y0, x1, y1))
                        # face = image[y0:y1, x0:x1]
                        faces.append(face)
                    detections_per_image.append(detections[i].shape[0])
                del img
                face_transforms = T.Compose([
                    T.Resize((65, 55)),
                    T.ToTensor(),
                    T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)
                ])

                # faces = [face_transforms(face) for face in faces]
                faces = torch.stack([face_transforms(face) for face in faces], dim=0)

                # print(faces.shape)

                logger.info('Got batch faces, starting inference on faces')
                with torch.no_grad():
                    faces = faces.to(device)
                    feat, _ = self.solider_model(faces)

                    evaluator.update((feat, timestamp, camid, trackid, imgpath, detections_per_image))

        logger.info('Starting evaluation')

        distmat, timestamps, camids, trackids, imgs_paths = evaluator.compute()
        return distmat, timestamps, camids, trackids, imgs_paths
