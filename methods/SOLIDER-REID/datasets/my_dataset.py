import glob
import logging
import os
import os.path as osp
import re
from datetime import datetime

from .bases import BaseImageDataset

EXTS = ['*.jpg', '*.png', '*.jpeg', '*.bmp', '*.ppm']


class MyDataset(BaseImageDataset):
    dataset_dir = 'my_dataset'

    def __init__(self, root="", verbose=True, day='both', **kwargs):
        super(MyDataset, self).__init__()
        self.day = day
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()
        self.query = self.__process_dir(self.query_dir)
        self.gallery = self.__process_dir(self.gallery_dir)

        if verbose:
            logger = logging.getLogger("transreid.dataset")
            logger.info('MyDataset has been loaded')
            self.print_custom_dataset_statistics(self.query, self.gallery)

        self.query_info = self.get_custom_image_data_info(self.query)
        self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_tracks \
            = self.get_custom_image_data_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise FileNotFoundError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.query_dir):
            raise FileNotFoundError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise FileNotFoundError("'{}' is not available".format(self.gallery_dir))

    def __process_dir(self, dir_path):
        if 'query' in dir_path:
            imgs_path = self.get_imgs_paths_query(dir_path)
        else:
            imgs_path = self.get_imgs_paths_gallery(dir_path, self.day)

        pattern = re.compile(r'SAT(?:_day\d+)?_Camera(\d+)_MILLIS_(\d+)_TRK-ID_(\d+)_TIMESTAMP_(\d+-\d+-\d+)')
        dataset = []
        for img_path in sorted(imgs_path):
            camid, _, trkid, timestamp = pattern.search(img_path).groups()
            camid = int(re.sub(r'_day\d+', '', camid))

            timestamp = datetime.strptime(timestamp, "%H-%M-%S").strftime("%H:%M:%S")
            dataset.append((img_path, timestamp, camid, trkid))

        return dataset

    @staticmethod
    def get_imgs_paths_query(dir_path):
        imgs_path = []
        for ext in EXTS:
            imgs_path.extend(glob.glob(osp.join(dir_path, ext)))

        return imgs_path

    @staticmethod
    def get_imgs_paths_gallery(dir_path, day):
        imgs_path = []
        for ext in EXTS:
            for root, _, _ in os.walk(dir_path):
                if not root.endswith(day) and day != 'both':
                    continue
                imgs_path.extend(glob.glob(osp.join(root, ext)))

        return imgs_path
