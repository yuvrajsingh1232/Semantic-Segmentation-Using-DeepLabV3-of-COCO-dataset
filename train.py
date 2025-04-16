import os
# os.environ["WANDB_MODE"] = "disabled"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import cv2
import torch
from pathlib import Path
import wandb
# wandb.init(project="segmentation_task", mode="offline")
import numpy as np
import argparse
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    MulticlassJaccardIndex as JaccardIndex
)
from torchvision.transforms import functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

# class CocoSegmentationDataset(Dataset):
#     def __init__(self, img_dir, mask_dir, transform=None, subset_size=5000):
#         self.img_dir = Path(img_dir)
#         self.mask_dir = Path(mask_dir)
#         self.transform = transform
        
#         # Get sorted list of image files
#         self.image_files = sorted(self.img_dir.glob("*.jpg"))[:subset_size]
#         self.mask_files = [self.mask_dir / f"{f.stem}.png" for f in self.image_files]

#         # Validate files
#         assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_path = self.image_files[idx]
#         mask_path = self.mask_files[idx]

#         # Load image and mask
#         image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
#         assert mask.max() < self.num_classes, f"Invalid mask value {mask.max()} in index {idx}"
#         assert mask.min() >= 0, f"Negative mask value in index {idx}"
          
#         if not isinstance(mask, np.ndarray):
#             mask = np.array(mask)
#         assert np.all((mask >= 0) & (mask < 255)), f"Mask has out-of-range values: {np.unique(mask)}"

#         # Handle overlapping masks or invalid annotations
#         mask = self.handle_overlapping_masks(mask)
# In CocoSegmentationDataset class definition
class CocoSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, subset_size=5000, num_classes=80):  # <- Add num_classes parameter
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.num_classes = num_classes  # <- Add num_classes attribute
        self.subset_size = subset_size

        # Get sorted list of image files
        self.image_files = sorted(self.img_dir.glob("*.jpg"))[:self.subset_size]
        self.mask_files = [self.mask_dir / f"{f.stem}.png" for f in self.image_files]

        # Validate files
        assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        # Load image and mask with validation
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Add null checks
        if image is None:
            raise ValueError(f"Failed to load image at {img_path}")
        if mask is None:
            raise ValueError(f"Failed to load mask at {mask_path}")
        
        # Convert to numpy array if needed
        mask = np.array(mask) if not isinstance(mask, np.ndarray) else mask
        
        # Clip values after validation
        mask = np.clip(mask, 0, self.num_classes-1)
        
        # Remaining code...
        
        # Move assertions after value correction
        assert mask.max() < self.num_classes, f"Invalid mask value {mask.max()} in index {idx}"
        assert mask.min() >= 0, f"Negative mask value in index {idx}"
               
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
  
        return image.float(), mask.long()

    def handle_overlapping_masks(self, mask):
        # Here, you can apply any overlap-handling strategy you prefer
        unique_values = np.unique(mask)
        if len(unique_values) > 1:
            largest_class = np.bincount(mask.flatten()).argmax()
            mask[mask != largest_class] = 0
        return mask

# class CocoSegmentationDataset(Dataset):
#     def __init__(self, img_dir, mask_dir, transform=None, subset_size=5000):
#         self.img_dir = Path(img_dir)
#         self.mask_dir = Path(mask_dir)
#         self.transform = transform
        
#         # Get sorted list of image files
#         self.image_files = sorted(self.img_dir.glob("*.jpg"))[:subset_size]
#         self.mask_files = [self.mask_dir / f"{f.stem}.png" for f in self.image_files]

#         # Validate files
#         assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_path = self.image_files[idx]
#         mask_path = self.mask_files[idx]

#         # Load image and mask
#         image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
          
#         if not isinstance(mask, np.ndarray):
#             mask = np.array(mask)
#         assert np.all((mask >= 0) & (mask < 255)), f"Mask has out-of-range values: {np.unique(mask)}"

#         if self.transform:
#             transformed = self.transform(image=image, mask=mask)
#             image = transformed['image']
#             mask = transformed['mask']
  
#         return image.float(), mask.long()

class DiceCoefficient:
    def __init__(self, num_classes: int, smooth: float = 1e-6):
        self.num_classes = num_classes
        self.smooth = smooth
        
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        y_pred = torch.argmax(y_pred, dim=1)  
        
        y_true = torch.nn.functional.one_hot(y_true, self.num_classes).permute(0, 3, 1, 2)  # (B, C, H, W)
        y_pred = torch.nn.functional.one_hot(y_pred, self.num_classes).permute(0, 3, 1, 2)  # (B, C, H, W)

        intersection = (y_pred * y_true).sum(dim=(2, 3))  # (B, C)
        union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))  # (B, C)

        dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)  # (B, C)

        return dice_per_class.mean().item()
    
class SegmentationModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Model architecture
        self.model = smp.DeepLabV3Plus(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=config['num_classes']
        )
        
        # Loss function
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(config['class_weights']),ignore_index=255)

        # Metrics
        self.train_iou = JaccardIndex(num_classes=config['num_classes'])
        self.val_iou = JaccardIndex(num_classes=config['num_classes'])
        self.dice = DiceCoefficient(num_classes=config['num_classes'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        outputs = self.model(inputs)

        self.pred = outputs
        self.target = targets
        loss = self.loss_fn(outputs, targets)
        
        return loss
        
    def on_train_epoch_end(self):
        pred = self.pred  
        target = self.target  
        dice_score = self.dice(pred, target)
        self.log("train_dice", dice_score, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        preds = torch.argmax(y_hat, dim=1)
        self.val_iou.update(preds, y)
        
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_iou", self.val_iou.compute(), prog_bar=True)
        self.val_iou.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_iou",
                "interval": "epoch",
                "frequency": 1
            }
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Train Segmentation Model')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Root directory of dataset')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for optimizer')
    parser.add_argument('--img_size', type=str, default="512,512",
                        help='Image size as "height,width"')
    return parser.parse_args()


def main(args):
    # Convert img_size to tuple
    img_size = tuple(map(int, args.img_size.split(',')))
    
    # Build config from arguments
    config = {
        "data_path": args.data_path,
        "num_classes": args.num_classes,
        "class_weights": [1.0] * args.num_classes, 
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.epochs,
        "img_size": img_size
    }

    # Data transformations
    train_transform = A.Compose([
        A.Resize(*config['img_size']),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(*config['img_size']),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # # Initialize dataset and dataloaders
    # train_dataset = CocoSegmentationDataset(
    #     img_dir=os.path.join(config['data_path'], "val2017"),
    #     mask_dir=os.path.join(config['data_path'], "masks"),
    #     transform=train_transform,
    #     subset_size=5000
    # )

    # val_dataset = CocoSegmentationDataset(
    #     img_dir=os.path.join(config['data_path'], "val2017"),
    #     mask_dir=os.path.join(config['data_path'], "masks"),
    #     transform=val_transform,
    #     subset_size=1000
    # )
    
    # In main() function where datasets are created
    # Update dataset initialization to include num_classes
    train_dataset = CocoSegmentationDataset(
        img_dir=os.path.join(config['data_path'], "images"),
        mask_dir=os.path.join(config['data_path'], "masks"),
        transform=train_transform,
        subset_size=4000,
        num_classes=config['num_classes']  # <- Add num_classes here
    )

    val_dataset = CocoSegmentationDataset(
        img_dir=os.path.join(config['data_path'], "images"),
        mask_dir=os.path.join(config['data_path'], "masks"),
        transform=val_transform,
        subset_size=800,
        num_classes=config['num_classes']  # <- And here
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=2
    )

    # Initialize model and trainer
    model = SegmentationModel(config)
    
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        logger=pl.loggers.WandbLogger(),
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_iou", patience=5, mode="max"),
            pl.callbacks.ModelCheckpoint(dirpath="checkpoints", monitor="val_iou", mode="max")
        ]
    )

    # Start training
    wandb.init(project="segmentation-training", config=config)
    trainer.fit(model, train_loader, val_loader)
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)

