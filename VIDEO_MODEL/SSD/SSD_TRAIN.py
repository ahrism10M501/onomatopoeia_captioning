# basic
import os
import glob
import json
import numpy as np
from PIL import Image

# torch
import torch
from torch import optim
import torchvision
from torch.utils.data import Dataset, DataLoader

# aug
import albumentations as A
from albumentations.pytorch import ToTensorV2

# util
from tqdm import tqdm

# custom
from data_collector import DataCollector

# Set Data Paths

root_path = ''

image_data = glob.glob(f"{root_path}\\train\\image\\**\\*.jpg", recursive=True)
image_annot_data = glob.glob(f"{root_path}\\train\\label\\**\\*.json", recursive=True)
val_image_data = glob.glob(f"{root_path}\\valid\\image\\**\\*.jpg", recursive=True)
val_image_annot_data = glob.glob(f"{root_path}\\valid\\label\\**\\*.json", recursive=True)

# Data Augmentations

train_transform = A.Compose([
    A.Resize(640, 640),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.OneOf([
        A.GaussianBlur(p=1),
        A.GaussNoise(p=1)
    ], p=0.5),
    A.Normalize(p=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_id'], min_visibility=0.0, clip=True)
)

valid_transform = A.Compose([
    A.Resize(640, 640),
    A.Normalize(p=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_id'], min_visibility=0.0, clip=True)
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

            # COCO format
            x_min = bbox["bndex_xcrdnt"]
            y_min = bbox["bndex_ycrdnt"]
            width = bbox["bndex_width"]
            height = bbox["bndex_hg"]

            boxes.append([x_min, y_min, x_min+width, y_min+height])
            labels.append(self.label_to_idx[meta['info']['class']['depth2']])

        image = Image.open(img_path).convert("RGB")
        # openCV는 한글 파일 경로를 읽지 못해 PIL 사용
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
    
# IoU 계산 함수
def compute_iou(box1, box2):
    """Compute IoU between two boxes (x1, y1, x2, y2)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0

# IoU 기준으로 prediction과 GT 매칭
def match_predictions(preds, gts, iou_threshold):
    matched = []
    used_gt = set()

    for pred_idx, pred_box in enumerate(preds['boxes']):
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gts['boxes']):
            if gt_idx in used_gt:
                continue
            iou = compute_iou(pred_box.tolist(), gt_box.tolist())
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            used_gt.add(best_gt_idx)
            matched.append((pred_idx, best_gt_idx))

    return matched

# precision, recall, f1 계산
def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1

# mAP, f1, acc 계산
def custom_map_eval(outputs, targets):
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    ap_list = []

    total_tp, total_fp, total_fn = 0, 0, 0
    total_correct = 0
    total_preds = 0

    for iou_thresh in iou_thresholds:
        all_tp, all_fp, all_fn = 0, 0, 0

        for pred, gt in zip(outputs, targets):
            matched = match_predictions(pred, gt, iou_thresh)

            tp = 0
            for pred_idx, gt_idx in matched:
                if pred['labels'][pred_idx] == gt['labels'][gt_idx]:
                    tp += 1

            fp = len(pred['boxes']) - tp
            fn = len(gt['boxes']) - tp

            all_tp += tp
            all_fp += fp
            all_fn += fn

            if iou_thresh == 0.5:
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_preds += len(pred['boxes'])
                total_correct += tp  # 정확도는 일치한 것만 세기

        precision = all_tp / (all_tp + all_fp + 1e-6)
        ap_list.append(precision)

    mAP_50_95 = np.mean(ap_list)
    mAP_50 = ap_list[0]

    acc = total_correct / (total_preds + 1e-6)
    precision, recall, f1 = precision_recall_f1(total_tp, total_fp, total_fn)

    return {
        'acc': acc,
        'f1': f1,
        'mAP': mAP_50_95,
        'mAP@.5': mAP_50
    }

# 최종 평가 함수
def evaluate(model, dataloader, device, conf_thresh=0.05):
    model.eval()
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = images.to(device)

            targets_list = []
            for i in range(len(targets['image_id'])):
                targets_list.append({
                    'image_id': targets['image_id'][i].unsqueeze(0).to(device),
                    'labels': targets['labels'][i].to(device),
                    'boxes': targets['boxes'][i].to(device)
                })

            outputs = model(images)

            for output, targ in zip(outputs, targets_list):
                scores = output['scores'].cpu()
                mask = scores > conf_thresh

                all_outputs.append({
                    'boxes': output['boxes'][mask].cpu(),
                    'labels': output['labels'][mask].cpu(),
                    'scores': scores[mask]
                })

                all_targets.append({
                    'boxes': targ['boxes'].cpu(),
                    'labels': targ['labels'].cpu()
                })

    result = custom_map_eval(all_outputs, all_targets)
    return result

# Main

if __name__ == '__main__':

    # Data preparation
    with open("SSD_hyper_param.json", 'r') as f:
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
    num_classes = len(train_dataset.label_to_idx) + 1

    # Model preparation
    model = torchvision.models.detection.ssd300_vgg16(num_classes=num_classes)

    try:
        if os.path.exists('SSD_best_model.pth'):
            model.load_state_dict(torch.load('SSD_best_model.pth'))
            print("SSD_best_model.pth loaded")
    except Exception as e:
        print(e)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Training is on a {device} environment")

    # Set Training parameters
    # cls_criterion = nn.CrossEntropyLoss()
    # bbox_criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=hyper_params["lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65, 90], gamma=0.1)

    num_epoch = hyper_params["epoch"]
    best_acc = buff["best_acc"]
    accumulation = hyper_params["accum"]

    # Train start
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(tqdm(train_dataloader)):

            # Processing Exception
            if targets['boxes'] is None or len(targets['boxes']) == 0:
              continue
            
            # inputs: [Batch, Channel, H, W]
            # targets: {'image_id':tensor(batch), 'labels':tensor(batch), 'boxes': tensor(batch,[x_min, y_min, width, height])}
            inputs = inputs.to(device)

            targets_list = []
            for i in range(len(targets['image_id'])):
                target = {
                    'image_id': targets['image_id'][i].unsqueeze(0).to(device),
                    'labels': targets['labels'][i].to(device),
                    'boxes': targets['boxes'][i].to(device)
                }
                targets_list.append(target)
            
            
            # model input: [Batch, Channel, H, W]
            # output: [[batch, num_classes], [batch, 4 (bbox)]]
            out_dict = model(inputs, targets_list)
            loss = sum(loss for loss in out_dict.values()) /accumulation
            loss.backward()
            

            if (i+1) % accumulation == 0 or (i + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

        scheduler.step()
        
        train_loss = running_loss / len(train_dataloader)

        # validating
        results = evaluate(model, val_dataloader, device_name)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
        print(f"Acc: {results['acc']:.4f} | f1: {results['f1']:.4f} | mAP: {results['mAP']:.4f} | mAP@0.5: {results['mAP@.5']:.4f}")
        for param_group in optimizer.param_groups:
            print("lr update:", param_group['lr'])
            
        # best pred model save
        if best_acc < results['acc']:
            best_acc = results['acc']
            torch.save(model.state_dict(), 'SSD_best_model.pth')
            print(f"Best model saved (Val Acc: {results['acc']:.4f})")

            buff['best_acc'] = best_acc
            with open("SSD_hyper_param.json", 'w') as f:
                json.dump(buff, f)
    
    # Train Accuracy
    tresults = evaluate(model, train_dataloader)
    print("epoch done")
    print(f"Acc: {tresults['acc']:.4f} | f1: {tresults['f1']:.4f} | mAP: {tresults['mAP']:.4f} | mAP@0.5: {tresults['mAP@.5']:.4f}")