import logging
import os.path as osp

import cv2
from PIL import Image, ImageFile
from torch.utils.data import Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path, use_cv2=False):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            if not use_cv2:
                img = Image.open(img_path).convert('RGB')
            else:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams = [], []

        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
            # tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        # tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        # num_views = len(tracks)
        return num_pids, num_imgs, num_cams  # , num_views

    def get_custom_image_data_info(self, data):
        cams, tracks = [], []

        for _, _, camid, trackid in data:
            cams += [camid]
            tracks += [trackid]
        cams = set(cams)
        tracks = set(tracks)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)
        logger = logging.getLogger("transreid.check")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # cameras")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        logger.info("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        logger.info("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        logger.info("  ----------------------------------------")

    def print_custom_dataset_statistics(self, query, gallery):
        num_query_imgs, num_query_cams, num_query_tracks = self.get_custom_image_data_info(query)
        num_gallery_imgs, num_gallery_cams, num_gallery_tracks = self.get_custom_image_data_info(gallery)
        logger = logging.getLogger("transreid.check")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # trk | # images | # cameras")
        logger.info("  ----------------------------------------")
        logger.info("  query    | {:5d} | {:8d} | {:9d}".format(num_query_tracks, num_query_imgs, num_query_cams))
        logger.info("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_tracks, num_gallery_imgs, num_gallery_cams))
        logger.info("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, custom=False, use_cv2=False):
        self.dataset = dataset
        self.transform = transform
        self.custom = custom
        self.use_cv2 = use_cv2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        if not self.custom:
            img_path, pid, camid = self.dataset[index]
            img = read_image(img_path, self.use_cv2)
            data_to_return = pid, camid, img_path
        else:
            img_path, timestamp, camid, trackid = self.dataset[index]
            img, img_path = (read_image(img_path, self.use_cv2), img_path) if isinstance(img_path, str) else (img_path, '')
            # img = read_image(img_path)
            data_to_return = timestamp, camid, trackid, img_path

        if self.transform is not None:
            img = self.transform(img)

        return img, *data_to_return
