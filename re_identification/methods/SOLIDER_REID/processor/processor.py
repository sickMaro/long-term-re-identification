import logging
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms as T
from torch.cuda import amp
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, CustomEvaluator

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from methods.FaceDetection_DSFD.utils import extract_faces
from my_utils.save_functions import save_state, load_state, save_batch_index, load_batch_index


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat, _ = model(img, label=target, cam_label=target_cam, view_label=target_view)
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else \
                            scheduler.get_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg,
                                            base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else \
                        scheduler.get_lr()[0]
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat, _ = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat, _ = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def move_models_to_device_and_eval_mode(models, device='cuda'):
    device = 'cpu' if device is None else device

    if torch.cuda.device_count() > 1:
        print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
        models = [nn.DataParallel(model) for model in models if model]

    for model in models:
        if model:
            model.to(device)
            model.eval()


def do_inference(cfg,
                 model,
                 face_detection_model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    start_dataset_name = cfg.TEST.WEIGHT.split('_')[-1].split('.')[0]
    if cfg.DATASETS.SPECIFIC_NAME:
        end_dataset_name = cfg.DATASETS.SPECIFIC_NAME
    else:
        end_dataset_name = cfg.DATASETS.NAMES

    if start_dataset_name == end_dataset_name:
        dataset_name = start_dataset_name
    else:
        dataset_name = 'specific_{}_to_{}'.format(
            start_dataset_name, end_dataset_name)
    test_type = cfg.OUTPUT_DIR.split('/')[-1]

    evaluator_path = 'evaluation_log/evaluation_state_{}_{}.pkl'.format(dataset_name, test_type)
    batch_index_path = 'evaluation_log/batch_index_{}_{}.pkl'.format(dataset_name, test_type)

    load_state(evaluator, logger, evaluator_path)
    start_idx = load_batch_index(logger, batch_index_path)

    evaluator.reranking = cfg.TEST.RE_RANKING

    move_models_to_device_and_eval_mode((model, face_detection_model), device)

    # img_path_list = []
    detections_per_image = []

    if start_idx < len(val_loader):
        for n_iter, (img, *batch_info) in enumerate(val_loader):
            if n_iter <= start_idx:
                continue  # Skip the batches we've already processed
            if face_detection_model:
                logger.info(f'Batch : {n_iter + 1}')
                logger.info(f'image batch: {img.shape}')
                result = get_faces(face_detection_model,
                                   (img, *batch_info),
                                   cfg,
                                   device,
                                   None)
                if result:
                    img, *batch_info, detections_per_image = result
                    logger.info('Got batch faces, starting inference on faces')
                else:
                    logger.info('No face detected')
                    continue

            logger.info(f'face batch: {img.shape}')
            with torch.no_grad():
                img = img.to(device)
                feat, _ = model(img)
                evaluator.update((feat, *batch_info[:-1], detections_per_image))
                # img_path_list.extend(batch_info[-1])

            if n_iter % 20 == 0:
                logger.info('Saving current state of evaluation...')
                save_state(evaluator, evaluator_path)
                save_batch_index(n_iter, batch_index_path)
            start_idx = n_iter

        logger.info('Saving final state of evaluation...')
        save_state(evaluator, evaluator_path)
        save_batch_index(start_idx + 1, batch_index_path)

    logger.info('Starting evaluation')
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.2%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


def do_custom_inference(cfg, model, face_detection_model, val_loader, num_query, query_from_gui):
    detections_per_image = []
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = CustomEvaluator(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)
    evaluator.reset()

    move_models_to_device_and_eval_mode((model, face_detection_model), device)

    # batch_info = timestamp, camid, trackid, img_path
    for n_iter, (img, *batch_info) in enumerate(val_loader):
        if face_detection_model:
            logger.info(f'Batch : {n_iter + 1}')
            logger.info(f'image batch: {img.shape}')
            result = get_faces(face_detection_model,
                               (img, *batch_info),
                               cfg,
                               device,
                               query_from_gui)
            if result:
                img, *batch_info, detections_per_image = result
                logger.info('Got batch faces, starting inference on faces')
            else:
                logger.info('No face detected')
                continue

        logger.info(f'face batch: {img.shape}')
        with torch.no_grad():
            img = img.to(device)
            feat, _ = model(img)
            evaluator.update((feat, *batch_info, detections_per_image))

    logger.info('Starting evaluation')
    distmat, timestamps, camids, trackids, imgs_paths = evaluator.compute()
    return distmat, timestamps, camids, trackids, imgs_paths


def get_faces(model, batch, cfg, device, query_from_gui):
    img, *batch_info, original_image = batch
    keep_index, faces, detections_per_image = model.detect_on_images_and_extract(img,
                                                                                 original_image,
                                                                                 device,
                                                                                 keep_thresh=0.7)

    if any(keep_index):
        original_image = np.array([np.array(image) for i, image in enumerate(original_image) if keep_index[i]], dtype=object)

        batch_info = [np.array(info)[keep_index] for info in batch_info]

        shape = (70, 60) if query_from_gui else (150, 150)
        face_transforms = T.Compose([
            T.Resize(shape),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])

        img = torch.stack([face_transforms(face) for face in faces], dim=0)

        return img, *batch_info, original_image, detections_per_image
    else:
        return None
