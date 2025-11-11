from config import dataset_config
from config import model_config
from config import train_config
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO: it does not work

from dataset import VOCDataset, download_and_extract, move_voc
import os
os.makedirs(dataset_config['BASE_DIR'], exist_ok=True)

download_and_extract(dataset_config['VOC2007_TRAINVAL_URL'], dataset_config['BASE_DIR'])
move_voc("VOC2007", "VOC2007")

download_and_extract(dataset_config['VOC2007_TEST_URL'], dataset_config['BASE_DIR'])
move_voc("VOC2007", "VOC2007-test")

download_and_extract(dataset_config['VOC2012_URL'], dataset_config['BASE_DIR'])
move_voc("VOC2012", "VOC2012")

# Cleanup VOCdevkit folder
vocdevkit_path = os.path.join(dataset_config['BASE_DIR'], "VOCdevkit")
if os.path.exists(vocdevkit_path) and not os.listdir(vocdevkit_path):
    os.rmdir(vocdevkit_path)
