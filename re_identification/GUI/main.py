import argparse
import os
import sys

from window import Window

sys.path.append('../methods/SOLIDER_REID')
from utils.logger import setup_logger
from config import cfg
from cameras_manager import VideoImageManager
from identification import ReIdentificationManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID on SAT-Cameras")
    parser.add_argument(
        "--config_file",
        default="../methods/SOLIDER_REID/configs/msmt17/swin_base.yml",
        help="path to config file",
        type=str)

    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
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

    video_manager = VideoImageManager()
    re_id_manager = ReIdentificationManager(cfg)
    re_id_manager.load_models()
    window = Window(cfg, video_manager=video_manager, re_id_manager=re_id_manager)
    window.show_window()
