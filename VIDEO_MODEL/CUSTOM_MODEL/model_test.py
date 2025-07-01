import os
import glob
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import albumentations as A
from albumentations.pytorch import ToTensorV2

import ObjectDetector
from data_collector import DataCollector
from model_train import CustomImageDataset 

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc,
    PrecisionRecallDisplay
)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = f'./explain/results_{timestamp}'
os.makedirs(save_dir, exist_ok=True)

# 테스트를 위한 데이터 준비
root_path = ''
image_data = glob.glob(f"{root_path}\\test\\image\\**\\*.jpg", recursive=True)
image_annot_data = glob.glob(f"{root_path}\\test\\label\\**\\*.json", recursive=True)

transform = A.Compose([
    A.PadIfNeeded(min_height=224),
    A.Resize(224, 224),
    A.Normalize(p=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

data = DataCollector(image_data, image_annot_data)
dataset = CustomImageDataset(data, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
print(dataset.label_to_idx)

# 모델 준비
model = ObjectDetector.ObjectDetector(num_classes=len(dataset.label_to_idx))
best_model_path = "best_model.pth"
model.to(device)

# best_loss.pth 모델 불러오기
try:
    model.load_state_dict(torch.load('best_model.pth'))
    print("best_acc.pth Loaded")
except Exception as e:
    print(e)

model.eval()

# 모델 테스트
correct = 0
total = 0

label_num = {'굽기': 0, '끓이기': 0, '냉장고 사용': 0, '드라이어 사용': 0, '믹서기 사용': 0, '썰기': 0, '압력밥솥 사용': 0, '전자레인지 사용': 0, '튀기기': 0}
label_true = {'굽기': 0, '끓이기': 0, '냉장고 사용': 0, '드라이어 사용': 0, '믹서기 사용': 0, '썰기': 0, '압력밥솥 사용': 0, '전자레인지 사용': 0, '튀기기': 0}
label_acc = {'굽기': 0, '끓이기': 0, '냉장고 사용': 0, '드라이어 사용': 0, '믹서기 사용': 0, '썰기': 0, '압력밥솥 사용': 0, '전자레인지 사용': 0, '튀기기': 0}

total_preds = []
total_labels = []
total_outputs = []

cls_mapping = { 0:'굽기', 1:'끓이기', 2:'냉장고 사용', 3:'드라이어 사용', 4:'믹서기 사용' , 5:'썰기', 6:'압력밥솥 사용', 7:'전자레인지 사용', 8:'튀기기'}
cls_mapping_en = { 0:'roasting', 1:'boiling', 2:'refrigerator', 3:'dryer', 4:'mixer' , 5:'chopping', 6:'pressure_cooker', 7:'microwave', 8:'frying'}

with torch.no_grad():
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        total_preds.append(preds.cpu())
        total_labels.append(labels.cpu())
        total_outputs.append(outputs.cpu())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for p, l in zip(preds, labels):
            for idx in range(4):
                if p.item() == l.item():
                    label_true[cls_mapping.get(l.item())] += 1
                label_num[cls_mapping.get(l.item())] += 1

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.4f}%")
   
    for k, v in label_num.items():
        label_acc[k] = round(label_true.get(k)/v, 3)

    print(label_acc)

total_preds = torch.cat(total_preds)
total_labels = torch.cat(total_labels)
total_outputs = torch.cat(total_outputs)

y_true = total_labels.cpu().numpy()
y_pred = total_preds.cpu().numpy()
probs = total_outputs.cpu().numpy()

class_names = list(cls_mapping_en.values())
num_classes = len(class_names)

# 1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
plt.close()

# 2. F1-score per class
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
f1_scores = [report[c]['f1-score'] for c in class_names]

plt.figure(figsize=(10, 6))
bars = plt.bar(class_names, f1_scores, color='orange')
for bar, score in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{score:.2f}', ha='center', va='bottom')
plt.ylim(0, 1)
plt.ylabel('F1-score')
plt.title('F1 Score per Class')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'f1_scores.png'))
plt.close()

# 3. Precision-Recall curve (macro)
plt.figure(figsize=(8, 6))
for i in range(num_classes):
    precision, recall, _ = precision_recall_curve(y_true == i, probs[:, i])
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot(ax=plt.gca(), name=f'{class_names[i]}')

plt.title('Precision-Recall Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'))
plt.close()

# 4. ROC Curve + AUC (macro)
plt.figure(figsize=(8, 6))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true == i, probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
plt.close()

print(f"All visualizations saved in: {save_dir}")