# basic
import os
import glob
import json
import numpy as np
from PIL import Image

# torch
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# aug
import albumentations as A
from albumentations.pytorch import ToTensorV2

# util
from tqdm import tqdm

# custom
import ObjectDetector
from data_collector import DataCollector

# Set Data Paths

root_path = ''

image_data = glob.glob(f"{root_path}\\train\\image\\**\\*.jpg", recursive=True)
image_annot_data = glob.glob(f"{root_path}\\train\\label\\**\\*.json", recursive=True)
val_image_data = glob.glob(f"{root_path}\\valid\\image\\**\\*.jpg", recursive=True)
val_image_annot_data = glob.glob(f"{root_path}\\valid\\label\\**\\*.json", recursive=True)

# Data Augmentations

train_transform = A.Compose([
    A.Resize(224, 224),
    A.PadIfNeeded(min_height=224, min_width=224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.OneOf([
        A.GaussianBlur(p=1),
        A.GaussNoise(p=1)
    ], p=0.5),
    A.Normalize(p=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='coco', label_fields=['category_id'], min_visibility=0.0, clip=True)
)

valid_transform = A.Compose([
    A.PadIfNeeded(min_height=224, min_width=224),
    A.Resize(224, 224),
    A.Normalize(p=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
],bbox_params=A.BboxParams(format='coco', label_fields=['category_id'], min_visibility=0.0, clip=True)
)

# Dataset

class CustomImageDataset(Dataset):
    def __init__(self, json_image_pairs, transform=None):
        self.data = json_image_pairs
        self.transform = transform

        # 라벨 매핑 (depth2 기준)
        self.labels = sorted(list(set([self._get_label(json_path) for _, json_path in self.data])))
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.data)
    
    def _get_label(self, json_path):
        with open(json_path, encoding='utf-8') as f:
            meta = json.load(f)
        return meta['info']['class']['depth2']
    
    def __getitem__(self, idx):
        img_path, json_path = self.data[idx]
        with open(json_path, encoding='utf-8') as f:
            meta = json.load(f)

        boxes = []
        labels = []

        for ann in meta['annotations']:
            bbox = ann["bbox"]
            if bbox is None or len(bbox) == 0:
                bbox = {"bndex_xcrdnt":0, "bndex_ycrdnt":0, "bndex_width":0, "bndex_hg":0}
                print("채워짐")

            x_min = bbox["bndex_xcrdnt"]
            y_min = bbox["bndex_ycrdnt"]
            width = bbox["bndex_width"]
            height = bbox["bndex_hg"]

            boxes.append([x_min, y_min, width, height])
            labels.append(self.label_to_idx[meta['info']['class']['depth2']])

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        if self.transform:
            augmentation = self.transform(image=image, bboxes=boxes, category_id=labels)
            image = augmentation['image']
            target['boxes'] = torch.tensor(augmentation['bboxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(augmentation['category_id'], dtype=torch.int64)

        else:
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.
            target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.tensor(labels, dtype=torch.int64)

        return image, target

# Main

if __name__ == '__main__':

    # Data preparation
    with open("hyper_param.json", 'r') as f:
        buff = json.load(f)
    hyper_params = buff["params"]

    # train_data: [[img_path, annot_path], ...]
    train_data = DataCollector(image_data, image_annot_data)
    val_data = DataCollector(val_image_data, val_image_annot_data)
    print(f"Train data size: {len(train_data)}, Valid data size: {len(val_data)}")

    # train_dataset: [image, target]
    # target -> {'image_id':int, 'labels':int, 'boxes': [x_min, y_min, width, height]}
    train_dataset = CustomImageDataset(train_data, transform=train_transform)
    val_dataset = CustomImageDataset(val_data, transform=valid_transform)
    print(f"Data_class: {train_dataset.label_to_idx}")

    # train_dataloader: [batch, channel, W, H]
    train_dataloader = DataLoader(train_dataset, batch_size=hyper_params["batch"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hyper_params["batch"], shuffle=False)

    # Model preparation
    model = ObjectDetector.ObjectDetector(num_classes=len(train_dataset.label_to_idx))

    try:
        if os.path.exists('best_model.pth'):
            model.load_state_dict(torch.load('best_model.pth'))
            print("best_model.pth loaded")
    except Exception as e:
        print(e)

     # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Training is on a {device} environment")
    # define valid function

    def evaluate(model, dataloader):
        model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader):

                # If you d like to check the data, refer to the training section
                inputs = inputs.to(device)
                labels = targets['labels'].to(device)
                bbox = targets['boxes'].to(device).squeeze()

                if len(labels) == 0 or len(bbox) == 0:
                    continue

                cls_pred, bbox_pred = model(inputs)

                cls_loss = cls_criterion(cls_pred.view(-1, model.num_classes), labels.view(-1))
                bbox_loss = bbox_criterion(bbox_pred, bbox)

                total_loss += cls_loss.item() + bbox_loss.item()
            
                cls_probs = torch.softmax(cls_pred, dim=1)
                _, cls_preds = torch.max(cls_probs, dim=1)

                correct += (cls_preds == labels).sum().item()
                total_samples += labels.numel()

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        acc = correct / total_samples if total_samples > 0 else 0

        return avg_loss, acc
    
    # Set Training parameters
    cls_criterion = nn.CrossEntropyLoss()
    bbox_criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=hyper_params["lr"])

    num_epoch = hyper_params["epoch"]
    best_acc = buff["best_acc"]
    
    # Train start
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0

        for inputs, targets in tqdm(train_dataloader):

            # Processing Exception
            if targets['boxes'] is None or len(targets['boxes']) == 0:
              continue
            
            # inputs: [Batch, Channel, H, W]
            # targets: [{'image_id':int, 'labels':int, 'boxes': [x_min, y_min, width, height]}, {}, ...]
            inputs, targets = inputs.to(device), targets
            
            cls_labels = targets['labels'].to(device)
            bbox_labels = targets['boxes'].to(device).squeeze()
            # Why squeeze? -> Target shape is [Batch, 1, bbox]
            # But the model expects input as [Batch, bbox]

            # model input: [Batch, Channel, H, W]
            # output: [[batch, num_classes], [batch, 4 (bbox)]]
            cls_pred, bbox_pred = model(inputs)

            # Calculate loss for both classification and bounding box
            cls_loss = cls_criterion(cls_pred.view(-1, model.num_classes), cls_labels.view(-1))
            bbox_loss = bbox_criterion(bbox_pred, bbox_labels)

            loss = cls_loss + bbox_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)

        # validating
        val_loss, val_acc = evaluate(model, val_dataloader)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
        print(f"Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        # best pred model save
        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved (Val Loss: {val_loss:.4f})")

            buff['best_acc'] = best_acc
            with open("hyper_param.json", 'w') as f:
                json.dump(buff, f)
    
    # Train Accuracy
    train_loss, train_acc = evaluate(model, train_dataloader)
    print("epoch done")
    print(f"Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")