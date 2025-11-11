from config import dataset_config
from config import model_config
from config import train_config
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from dataset import VOCDataset
from torch.utils.data import DataLoader

from dataset import VOCDataset
from torch.utils.data import DataLoader

def collate_function(data):
    return tuple(zip(*data))

train_dataset = VOCDataset('train',
                    im_sets=dataset_config['train_im_sets'],
                    im_size=dataset_config['im_size'])

test_dataset = VOCDataset('test',
                    im_sets=dataset_config['test_im_sets'],
                    im_size=dataset_config['im_size'])

train_dataloader = DataLoader(train_dataset,
                            batch_size=train_config['batch_size'],
                            shuffle=True,
                            collate_fn=collate_function,
                            #num_workers=dataset_config['num_workers'],
                            pin_memory=True
                            )

test_dataloader = DataLoader(test_dataset,
                            batch_size=train_config['batch_size'],
                            shuffle=True,
                            collate_fn=collate_function,
                            #num_workers=dataset_config['num_workers'],
                            pin_memory=True
                            )

print(f"Train data size: {len(train_dataset)}, Test data size: {len(test_dataset)}")
print(f"Train dataloader size: {len(train_dataloader)}, Test dataloader size: {len(test_dataloader)}")