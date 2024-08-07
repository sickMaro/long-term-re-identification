#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import sys

sys.path.append('..')
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.evaluation import print_csv_format
from collections import OrderedDict
import logging


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args, results):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
        res = DefaultTrainer.test(cfg, model)

        results[cfg.DATASETS.KWARGS.split(':')[1]] = res
        return res

    trainer = DefaultTrainer(cfg)
    # load trained model to funetune
    if args.finetune:
        Checkpointer(trainer.model).load(cfg.MODEL.WEIGHTS)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def custom_set(args, rerank=False, config_file='../configs/CMDM/mgn_R50_moco.yml'):
    args.eval_only = True
    args.resume = False
    args.config_file = config_file
    datasets = args.opts
    base_opts = ['DATASETS.ROOT', '../../../../datasets', 'TEST.RERANK.ENABLED', rerank,
                 'TEST.AQE.ENABLED', False, 'TEST.ROC_ENABLED', False]
    results = OrderedDict()
    logger = logging.getLogger(__name__)
    for dataset in datasets:
        args.opts = base_opts + [
            'DATASETS.KWARGS', f'data_name:{dataset}',
            'MODEL.WEIGHTS', f'../../pre_trained_models_path/market.pth',
            'OUTPUT_DIR', f'../logs/lup_moco/test/market_to_{dataset}'
        ]
        logger.info(f"Custom script Args for dataset: {dataset} ", args)
        main(args, results)
        '''launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args, results),
        )'''
    logger.info('Print test results: ')
    print_csv_format(results)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    if args.config_file:
        print("Command Line Args:", args)
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
    else:
        custom_set(args, rerank=True)
