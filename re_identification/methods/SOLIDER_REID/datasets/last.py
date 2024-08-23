# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os
import os.path as osp
import re

from .bases import BaseImageDataset

EXTS = ['*.jpg', '*.png', '*.jpeg', '*.bmp', '*.ppm']


class LaST(BaseImageDataset):
    dataset_dir = 'last'

    def __init__(self, root='', verbose=True, **kwargs):
        super(LaST, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'test', 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test', 'gallery')

        self._check_before_run()
        train = self._process_dir(self.train_dir)
        query = self._process_dir(self.query_dir)
        gallery = self._process_dir(self.gallery_dir, recam=len(query))

        if verbose:
            print("=> LaST loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, recam=0):
        if 'query' in dir_path:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        else:
            img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))

        img_paths = sorted(img_paths)
        dataset = []
        for ii, img_path in enumerate(img_paths):
            pid = int(os.path.basename(img_path).split('_')[0])
            camid = int(recam + ii)
            dataset.append((img_path, pid, camid))

        return dataset
