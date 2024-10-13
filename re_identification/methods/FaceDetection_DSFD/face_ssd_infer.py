import sys

import cv2
import torch
import torchvision
import torch.nn as nn
import numpy as np
from PIL import Image

from data.config import TestBaseTransform, widerface_640 as cfg
from layers import Detect, get_prior_boxes, FEM, pa_multibox, mio_module, upsample_product


class SSD(nn.Module):

    def __init__(self, phase, nms_thresh=0.3, nms_conf_thresh=0.01):
        super(SSD, self).__init__()

        self.phase = phase
        self.num_classes = 2
        self.cfg = cfg
        self.priors = None

        resnet = torchvision.models.resnet152()

        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)
        self.layer5 = nn.Sequential(
            *[nn.Conv2d(2048, 512, kernel_size=1),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True),
              nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True)]
        )
        self.layer6 = nn.Sequential(
            *[nn.Conv2d(512, 128, kernel_size=1, ),
              nn.BatchNorm2d(128),
              nn.ReLU(inplace=True),
              nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
              nn.BatchNorm2d(256),
              nn.ReLU(inplace=True)]
        )

        output_channels = [256, 512, 1024, 2048, 512, 256]

        # FPN
        fpn_in = output_channels

        self.latlayer3 = nn.Conv2d(fpn_in[3], fpn_in[2], kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(fpn_in[2], fpn_in[1], kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(fpn_in[1], fpn_in[0], kernel_size=1, stride=1, padding=0)

        self.smooth3 = nn.Conv2d(fpn_in[2], fpn_in[2], kernel_size=1, stride=1, padding=0)
        self.smooth2 = nn.Conv2d(fpn_in[1], fpn_in[1], kernel_size=1, stride=1, padding=0)
        self.smooth1 = nn.Conv2d(fpn_in[0], fpn_in[0], kernel_size=1, stride=1, padding=0)

        # FEM
        cpm_in = output_channels

        self.cpm3_3 = FEM(cpm_in[0])
        self.cpm4_3 = FEM(cpm_in[1])
        self.cpm5_3 = FEM(cpm_in[2])
        self.cpm7 = FEM(cpm_in[3])
        self.cpm6_2 = FEM(cpm_in[4])
        self.cpm7_2 = FEM(cpm_in[5])

        # head
        head = pa_multibox(output_channels)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)

        if self.phase != 'onnx_export':
            self.detect = Detect(self.num_classes, 0, cfg['num_thresh'], nms_conf_thresh, nms_thresh,
                                 cfg['variance'])
            self.last_image_size = None
            self.last_feature_maps = None

        if self.phase == 'test':
            self.test_transform = TestBaseTransform((104, 117, 123))

    def forward(self, x):

        image_size = [x.shape[2], x.shape[3]]
        loc = list()
        conf = list()

        conv3_3_x = self.layer1(x)
        conv4_3_x = self.layer2(conv3_3_x)
        conv5_3_x = self.layer3(conv4_3_x)
        fc7_x = self.layer4(conv5_3_x)
        conv6_2_x = self.layer5(fc7_x)
        conv7_2_x = self.layer6(conv6_2_x)

        lfpn3 = upsample_product(self.latlayer3(fc7_x), self.smooth3(conv5_3_x))
        lfpn2 = upsample_product(self.latlayer2(lfpn3), self.smooth2(conv4_3_x))
        lfpn1 = upsample_product(self.latlayer1(lfpn2), self.smooth1(conv3_3_x))

        conv5_3_x = lfpn3
        conv4_3_x = lfpn2
        conv3_3_x = lfpn1

        sources = [conv3_3_x, conv4_3_x, conv5_3_x, fc7_x, conv6_2_x, conv7_2_x]

        sources[0] = self.cpm3_3(sources[0])
        sources[1] = self.cpm4_3(sources[1])
        sources[2] = self.cpm5_3(sources[2])
        sources[3] = self.cpm7(sources[3])
        sources[4] = self.cpm6_2(sources[4])
        sources[5] = self.cpm7_2(sources[5])

        # apply multibox head to source layers
        featuremap_size = []
        for (x, l, c) in zip(sources, self.loc, self.conf):
            featuremap_size.append([x.shape[2], x.shape[3]])
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            len_conf = len(conf)
            cls = mio_module(c(x), len_conf)
            conf.append(cls.permute(0, 2, 3, 1).contiguous())

        face_loc = torch.cat([o[:, :, :, :4].contiguous().view(o.size(0), -1) for o in loc], 1)
        face_loc = face_loc.view(face_loc.size(0), -1, 4)
        face_conf = torch.cat([o[:, :, :, :2].contiguous().view(o.size(0), -1) for o in conf], 1)
        face_conf = self.softmax(face_conf.view(face_conf.size(0), -1, self.num_classes))

        if self.phase != 'onnx_export':

            if (self.last_image_size is None or self.last_image_size != image_size or
                    self.last_feature_maps != featuremap_size):
                self.priors = get_prior_boxes(self.cfg, featuremap_size, image_size).to(face_loc.device)
                self.last_image_size = image_size
                self.last_feature_maps = featuremap_size
            with torch.no_grad():
                output = self.detect(face_loc, face_conf, self.priors)
        else:
            output = torch.cat((face_loc, face_conf), 2)
        return output

    def detect_on_images(self, images, scales, device, keep_thresh=0.3):

        x = images.to(device)
        detections = self.forward(x).cpu().numpy()

        filter_list = np.zeros((detections.shape[0]), dtype=object)
        for i in range(detections.shape[0]):
            scores = detections[i, 1, :, 0]
            keep_idxs = scores > keep_thresh  # find keeping indexes

            filter_list[i] = detections[i, 1, keep_idxs, :]  # select over threshold
            filter_list[i] = filter_list[i][:, [1, 2, 3, 4, 0]]  # reorder
            filter_list[i][:, :4] *= scales[i]

            for bbox in filter_list[i]:
                x0, y0, x1, y1 = bbox[:4]

                multiplier_x = 0.7 if (x1 - x0) <= 33 else 0.5
                multiplier_y = 0.5 if (y1 - y0) <= 33 else 0.4

                bbox[0] = max(0, bbox[0] - ((x1 - x0) * multiplier_x))
                bbox[1] = max(0, bbox[1] - ((y1 - y0) * multiplier_y))
                bbox[2] = min(scales[i][0], bbox[2] + ((x1 - x0) * multiplier_x))
                bbox[3] = min(scales[i][1], bbox[3] + ((y1 - y0) * multiplier_y))

        return filter_list

    def detect_on_images_and_extract(self, images, original_images, device, keep_thresh=0.3):

        x = images.to(device)
        detections = self.forward(x).cpu().numpy()

        faces = []
        detections_per_image = []
        batch_info_to_keep = []
        target_size = images[0].shape[1:]
        print(target_size)
        for i in range(detections.shape[0]):
            scores = detections[i, 1, :, 0]
            keep_idxs = scores > keep_thresh  # find keeping indexes
            not_skip = any(keep_idxs)
            batch_info_to_keep.append(not_skip)

            if not_skip:
                det = detections[i, 1, keep_idxs, :]  # select over threshold
                det = det[:, [1, 2, 3, 4, 0]]  # reorder
                # scale = (*original_images[i].size, *original_images[i].size)
                curr_img_size = original_images[i].size[::-1]
                print(curr_img_size)
                resize_factor_x = target_size[1] / curr_img_size[1]
                resize_factor_y = target_size[0] / curr_img_size[0]

                scale = (target_size[1] / resize_factor_x,
                         target_size[0] / resize_factor_y,
                         target_size[1] / resize_factor_x,
                         target_size[0] / resize_factor_y)

                det[:, :4] *= scale

                for bbox in det:
                    x0, y0, x1, y1 = bbox[:4]

                    # Compute multipliers
                    multiplier_x = 0.7 if (x1 - x0) <= 33 else 0.5
                    multiplier_y = 0.5 if (y1 - y0) <= 33 else 0.4

                    # Expand bbox
                    x0 = int(max(0, x0 - (x1 - x0) * multiplier_x))
                    y0 = int(max(0, y0 - (y1 - y0) * multiplier_y))
                    x1 = int(min(scale[0], x1 + (x1 - x0) * multiplier_x))
                    y1 = int(min(scale[1], y1 + (y1 - y0) * multiplier_y))

                    # mascheramento area intorno al volto
                    # start_img = np.array(original_images[i])
                    # blurred_image = cv2.GaussianBlur(start_img, (33, 33), 0)
                    # blurred_image[y0:y1, x0:x1] = start_img[y0:y1, x0:x1]
                    # faces.append((Image.fromarray(blurred_image)))

                    # crop del volto
                    faces.append(original_images[i].crop((x0, y0, x1, y1)))

                detections_per_image.append(det.shape[0])

        return batch_info_to_keep, faces, detections_per_image
