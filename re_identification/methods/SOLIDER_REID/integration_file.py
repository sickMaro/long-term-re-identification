import argparse
import os

from config import cfg
from datasets.make_dataloader import make_custom_dataloader
from model import make_model
from processor import do_custom_inference
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/msmt17/swin_base.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

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

    return cfg


def get_model(config):
    model = make_model(config, num_class=0, camera_num=0, view_num=0,
                       semantic_weight=config.MODEL.SEMANTIC_WEIGHT)

    model.load_param(config.TEST.WEIGHT)

    return model


def get_inference_results(config, model):
    val_loader, num_query, _, _ = make_custom_dataloader(config)

    return do_custom_inference(config, model, val_loader, num_query)


if __name__ == "__main__":
    cfg = parse_args()
    get_inference_results(cfg, get_model(cfg))
