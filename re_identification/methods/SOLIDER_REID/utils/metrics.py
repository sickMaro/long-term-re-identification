import logging
import sys

import numpy as np
import torch
from utils.reranking import re_ranking


def get_best_matrix(distmat, det_for_image, num_query):
    if len(det_for_image) > 0:
        query_detections = det_for_image[:num_query]
        gallery_detection = det_for_image[num_query:]
        best_rows = np.zeros((len(query_detections), distmat.shape[1]))

        start = 0
        for i, num_det in enumerate(query_detections):
            end = start + num_det
            best_rows[i] = np.min(distmat[start:end], axis=0)
            start = end

        best_dist_matrix = np.zeros((len(query_detections), len(gallery_detection)))

        start = 0
        for i, num_det in enumerate(gallery_detection):
            end = start + num_det
            best_dist_matrix[:, i] = np.min(best_rows[:, start:end], axis=1)
            start = end

    else:
        best_dist_matrix = distmat
    return best_dist_matrix


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # dist_mat.addmm_(1, -2, qf, gf.t())
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval:
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.detections_per_image = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, detections_per_image = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.detections_per_image.extend(np.asarray(detections_per_image))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        # qf = feats[:self.num_query]
        qf = feats[:sum(self.detections_per_image[:self.num_query])]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        # gf = feats[self.num_query:]
        gf = feats[sum(self.detections_per_image[:self.num_query]):]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        if self.reranking:
            print('=> Enter reranking')
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)

        best_dist = get_best_matrix(distmat, self.detections_per_image, self.num_query)
        print(best_dist.shape)
        cmc, mAP = eval_func(best_dist, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, best_dist, self.pids, self.camids, qf, gf


class CustomEvaluator:
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(CustomEvaluator, self).__init__()
        self.det_for_image = []
        self.imgs_paths = []
        self.trackids = []
        self.camids = []
        self.timestamps = []
        self.feats = []
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.timestamps = []
        self.camids = []
        self.trackids = []
        self.imgs_paths = []
        self.det_for_image = []

    def update(self, output):  # called once for each batch
        feat, timsetamp, camid, trackid, img_path, det_for_image = output
        self.feats.append(feat.cpu())
        self.timestamps.extend(np.asarray(timsetamp))
        self.camids.extend(np.asarray(camid))
        self.trackids.extend(np.asarray(trackid))
        self.imgs_paths.extend(np.asarray(img_path))
        self.det_for_image.extend(np.asarray(det_for_image))

    def compute(self):  # called after each epoch
        logger = logging.getLogger('transreid.evaluate')
        feats = torch.cat(self.feats, dim=0)

        if self.feat_norm:
            logger.info("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]

        # gallery
        gf = feats[self.num_query:]

        if self.reranking:
            logger.info('=> Enter reranking')
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)

        else:
            logger.info('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)

        print(distmat.shape)

        best_dist = get_best_matrix(distmat, self.det_for_image, self.num_query)

        indixes = np.squeeze(np.argsort(best_dist, axis=1))
        self.timestamps = (np.array(self.timestamps[self.num_query:]))[indixes]
        self.camids = (np.array(self.camids[self.num_query:]))[indixes]
        self.trackids = (np.array(self.trackids[self.num_query:]))[indixes]
        self.imgs_paths = (np.array(self.imgs_paths[self.num_query:]))[indixes]

        return (best_dist,
                self.timestamps,
                self.camids,
                self.trackids,
                self.imgs_paths)
