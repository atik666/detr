from config import dataset_config
from config import model_config
from config import train_config
import torch
from dataset import VOCDataset
from torch.utils.data import DataLoader

from dataset import VOCDataset
from torch.utils.data import DataLoader
from model import *
from torchinfo import summary
from torch.optim.lr_scheduler import MultiStepLR
from loss import *
from train import train
import matplotlib.pyplot as plt
from visualization import visualize_detr_output

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

test = torch.randn((16, 3, 640, 640))

model_detr = DETR().to(device)
test_output = model_detr(test.to(device))
print(test_output['pred_logits'].shape)  # Should be (num_decoders, B, num_queries, num_classes)
print(test_output['pred_boxes'].shape)  # Should be (num_decoders, B, num_queries, 4)

summary(model=model_detr, 
        input_size=(train_config['batch_size'], 3, dataset_config['im_size'], dataset_config['im_size']),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
) 

loss_fn = DETRLoss()
optimizer = torch.optim.AdamW(
    params=filter(lambda p: p.requires_grad, model_detr.parameters()),
    lr=train_config['lr'],
    weight_decay=1e-4)

lr_scheduler = MultiStepLR(optimizer,
                            train_config['lr_steps'],
                            gamma=0.1)

loss_history = train(model=model_detr,
          dataloader=train_dataloader,
          device=device,
          loss_fn=loss_fn,
          optimizer=optimizer,
          scheduler=lr_scheduler,
          epochs=train_config['num_epochs'])

plt.plot(loss_history['classification'], label='Classification Loss')
plt.xlabel('Epochs')
plt.ylabel('Class Loss')
plt.title('Loss History')
plt.legend()
plt.show()

plt.plot(loss_history['bbox_regression'], label='Bbox Regression Loss')
plt.xlabel('Epochs')
plt.ylabel('Bbox Loss')
plt.title('Localization Loss History')
plt.legend()
plt.show()

visualize_detr_output(
    model=model_detr,
    dataloader=test_dataloader,
    loss_fn=DETRLoss(),
    device=device,
    batch_size=4,
    score_thresh=0.7
)