echo 'swim_base'
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/market/swin_base.yml DATASETS.SPECIFIC_NAME 'msmt17' TEST.WEIGHT './pretrained_models/swin_base_market.pth' TEST.RE_RANKING False MODEL.SEMANTIC_WEIGHT 0.2
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/market/swin_base.yml DATASETS.SPECIFIC_NAME 'cuhk03' TEST.WEIGHT './pretrained_models/swin_base_market.pth' TEST.RE_RANKING True MODEL.SEMANTIC_WEIGHT 0.2
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/market/swin_base.yml DATASETS.SPECIFIC_NAME 'duke' TEST.WEIGHT './pretrained_models/swin_base_market.pth' TEST.RE_RANKING True MODEL.SEMANTIC_WEIGHT 0.2


CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/msmt17/swin_base.yml DATASETS.SPECIFIC_NAME 'market1501' TEST.WEIGHT './pretrained_models/swin_base_msmt17.pth' TEST.RE_RANKING True MODEL.SEMANTIC_WEIGHT 0.2
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/msmt17/swin_base.yml DATASETS.SPECIFIC_NAME 'cuhk03' TEST.WEIGHT './pretrained_models/swin_base_msmt17.pth' TEST.RE_RANKING True MODEL.SEMANTIC_WEIGHT 0.2
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/msmt17/swin_base.yml DATASETS.SPECIFIC_NAME 'duke' TEST.WEIGHT './pretrained_models/swin_base_msmt17.pth' TEST.RE_RANKING True MODEL.SEMANTIC_WEIGHT 0.2


read -p 'Press [Enter] to exit...'