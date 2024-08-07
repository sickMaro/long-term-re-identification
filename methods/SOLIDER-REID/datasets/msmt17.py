import os
import os.path as osp
import re
from glob import glob

from .bases import BaseImageDataset

EXTS = ['*.jpg', '*.png', '*.jpeg', '*.bmp', '*.ppm']


class MSMT17(BaseImageDataset):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = 'msmt17'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(MSMT17, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=False)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

            self.train = train
            self.query = query
            self.gallery = gallery

            self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = (
                self.get_imagedata_info(self.train))

            self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = (
                self.get_imagedata_info(self.query))

            self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = \
                self.get_imagedata_info(self.gallery)

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

    def _process_dir(self, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c(\d+)')
        all_pids, all_cids = {}, {}
        ret, fpaths = [], []
        for ext in EXTS:
            fpaths.extend(glob(os.path.join(path, ext)))
        fpaths = sorted(fpaths)
        for fpath in fpaths:
            fname = os.path.basename(fpath)
            pid, cid = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            if cid not in all_cids:
                all_cids[cid] = cid
            pid = all_pids[pid]
            cid -= 1
            ret.append((fpath, pid, cid, 0))
        return ret
