CUDA_VISIBLE_DEVICES=0 python test_clipreid.py --config_file configs/person/vit_clipreid.yml DATASETS.SPECIFIC_NAME 'duke' TEST.WEIGHT 'pretrained_models/vit/Market1501_clipreid_ViT-B-16_60.pth'
CUDA_VISIBLE_DEVICES=0 python test_clipreid.py --config_file configs/person/vit_clipreid.yml DATASETS.SPECIFIC_NAME 'cuhk03' TEST.WEIGHT 'pretrained_models/vit/Market1501_clipreid_ViT-B-16_60.pth'
CUDA_VISIBLE_DEVICES=0 python test_clipreid.py --config_file configs/person/vit_clipreid.yml DATASETS.SPECIFIC_NAME 'msmt17' TEST.WEIGHT 'pretrained_models/vit/Market1501_clipreid_ViT-B-16_60.pth'


CUDA_VISIBLE_DEVICES=0 python test_clipreid.py --config_file configs/person/vit_clipreid.yml DATASETS.SPECIFIC_NAME 'duke' TEST.WEIGHT 'pretrained_models/vit/MSMT17_clipreid_ViT-B-16_60.pth'
CUDA_VISIBLE_DEVICES=0 python test_clipreid.py --config_file configs/person/vit_clipreid.yml DATASETS.SPECIFIC_NAME 'cuhk03' TEST.WEIGHT 'pretrained_models/vit/MSMT17_clipreid_ViT-B-16_60.pth'
CUDA_VISIBLE_DEVICES=0 python test_clipreid.py --config_file configs/person/vit_clipreid.yml DATASETS.SPECIFIC_NAME 'market1501' TEST.WEIGHT 'pretrained_models/vit/MSMT17_clipreid_ViT-B-16_60.pth'

read -p 'Press [Enter] ...'