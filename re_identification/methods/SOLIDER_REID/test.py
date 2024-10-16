import argparse
import os

import torch
import torchvision.transforms as T
from config import cfg
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
import sys
sys.path.append('../methods/FaceDetection_DSFD')
sys.path.append('../FaceDetection_DSFD')
from face_ssd_infer import SSD
from datasets.transformation import CustomTransform

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    start_dataset_name = cfg.TEST.WEIGHT.split('_')[-1].split('.')[0]
    end_dataset_name = cfg.DATASETS.SPECIFIC_NAME

    if start_dataset_name != end_dataset_name:
        output_dir = './log/cross_dataset/{}_to_{}/'.format(start_dataset_name,
                                                            end_dataset_name)
    else:
        output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    if cfg.TEST.USE_FACE_DETECTION:
        face_detection_model = SSD("test")
        path = '../FaceDetection_DSFD/pretrained_models/WIDERFace_DSFD_RES152.pth'
        face_detection_model.load_state_dict(torch.load(path))

        custom_t = CustomTransform(cfg.INPUT.SIZE_TEST, face_detection_model.test_transform)
        val_transforms = T.Compose([custom_t])
    else:
        face_detection_model = None
        val_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])

    (train_loader, train_loader_normal,
     val_loader, num_query, num_classes, camera_num) = make_dataloader(cfg, val_transforms)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=0,
                       semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)

    if cfg.TEST.WEIGHT != '':
        model.load_param(cfg.TEST.WEIGHT)


    if cfg.DATASETS.NAMES == 'VehicleID':
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(
                cfg)
            rank_1, rank5 = do_inference(cfg,
                                         model,
                                         val_loader,
                                         num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum() / 10.0, all_rank_5.sum() / 10.0))
    else:
        do_inference(cfg,
                     model,
                     face_detection_model,
                     val_loader,
                     num_query)
