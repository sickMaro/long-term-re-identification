echo 'swin_small'
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/market/swin_base.yml TEST.WEIGHT './pretrained_models/swin_base_market.pth' TEST.RE_RANKING False MODEL.SEMANTIC_WEIGHT 0.2
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/market/swin_base.yml TEST.WEIGHT './pretrained_models/swin_base_market.pth' TEST.RE_RANKING True MODEL.SEMANTIC_WEIGHT 0.2
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/msmt17/swin_base.yml TEST.WEIGHT './pretrained_models/swin_base_msmt17.pth' TEST.RE_RANKING False MODEL.SEMANTIC_WEIGHT 0.2

echo 'swin_small'
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/market/swin_small.yml TEST.WEIGHT './pretrained_models/swin_small_market.pth' TEST.RE_RANKING False MODEL.SEMANTIC_WEIGHT 0.2
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/market/swin_small.yml TEST.WEIGHT './pretrained_models/swin_small_market.pth' TEST.RE_RANKING True MODEL.SEMANTIC_WEIGHT 0.2
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/msmt17/swin_small.yml TEST.WEIGHT './pretrained_models/swin_small_msmt17.pth' TEST.RE_RANKING False MODEL.SEMANTIC_WEIGHT 0.2

echo 'swin_tiny'
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/market/swin_tiny.yml TEST.WEIGHT './pretrained_models/swin_tiny_market.pth' TEST.RE_RANKING False MODEL.SEMANTIC_WEIGHT 0.2
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/market/swin_tiny.yml TEST.WEIGHT './pretrained_models/swin_tiny_market.pth' TEST.RE_RANKING True MODEL.SEMANTIC_WEIGHT 0.2
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/msmt17/swin_tiny.yml TEST.WEIGHT './pretrained_models/swin_tiny_msmt17.pth' TEST.RE_RANKING False MODEL.SEMANTIC_WEIGHT 0.2
