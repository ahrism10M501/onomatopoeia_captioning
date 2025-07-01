import os
import json
import glob
from tqdm import tqdm
import numpy as np
from datetime import datetime

import torch
import torchaudio
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import Audio_labeler
from data_collector import DataCollector
from TRAIN import CustomSoundDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = f'./audio_runs/results_{timestamp}'
os.makedirs(save_dir, exist_ok=True)

# 테스트 데이터 경로 설정
root_path = ''
sound_data = glob.glob(f"{root_path}\\test\\audio\\**\\*.wav", recursive=True)
sound_annot_data = glob.glob(f"{root_path}\\test\\label\\**\\*.json", recursive=True)

data = DataCollector(sound_data, sound_annot_data)
dataset = CustomSoundDataset(data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# 모델 불러오기
model = Audio_labeler.SoundClassifier(num_classes=len(dataset.label_to_idx), num_oto=9)
model_path = 'best_model2.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

correct = 0
total = 0

label_num = {'굽기': 0, '끓이기': 0, '냉장고 사용': 0, '드라이어 사용': 0, '믹서기 사용': 0, '썰기': 0, '압력밥솥 사용': 0, '전자레인지 사용': 0, '튀기기': 0}
label_true = {'굽기': 0, '끓이기': 0, '냉장고 사용': 0, '드라이어 사용': 0, '믹서기 사용': 0, '썰기': 0, '압력밥솥 사용': 0, '전자레인지 사용': 0, '튀기기': 0}
label_acc = {'굽기': 0, '끓이기': 0, '냉장고 사용': 0, '드라이어 사용': 0, '믹서기 사용': 0, '썰기': 0, '압력밥솥 사용': 0, '전자레인지 사용': 0, '튀기기': 0}
# 라벨별 개수 및 정확도 계산용 딕셔너리
# 유연하게 대처하도록 만들어야하는데 실수

total_preds = []
total_labels = []
total_outputs = []
# 예측결과 저장용 리스트

cls_mapping = { 0:'굽기', 1:'끓이기', 2:'냉장고 사용', 3:'드라이어 사용', 4:'믹서기 사용' , 5:'썰기', 6:'압력밥솥 사용', 7:'전자레인지 사용', 8:'튀기기'}
cls_mapping_en = { 0:'roasting', 1:'boiling', 2:'refrigerator', 3:'dryer', 4:'mixer' , 5:'chopping', 6:'pressure_cooker', 7:'microwave', 8:'frying'}
# 예측값을 사람이 알아볼 수 있게 출력하거나, 결과 저장 시 활용

# 예측 결과 저장용
true_labels = []
pred_labels = []
true_onos = []
pred_onos = []

# 평가 수행
with torch.no_grad():
    for inputs, labels, otos in tqdm(dataloader, desc="Testing"):
        inputs, labels, otos = inputs.to(device), labels.to(device), otos.to(device)
        label_out, ono_out = model(inputs)

        _, label_preds = torch.max(label_out, 1)
        _, ono_preds = torch.max(ono_out, 1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(label_preds.cpu().numpy())
        true_onos.extend(otos.cpu().numpy())
        pred_onos.extend(ono_preds.cpu().numpy())

# 라벨 이름 매핑
label_names = list(dataset.label_to_idx.keys())
ono_names = list(dataset.ono_to_idx.keys())

# confusion matrix 시각화 함수
def plot_confusion_matrix(y_true, y_pred, classes, title, file_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()

# plot confusion matrix for both
plot_confusion_matrix(true_labels, pred_labels, label_names, "Label Confusion Matrix", "label_conf_matrix.png")
plot_confusion_matrix(true_onos, pred_onos, ono_names, "Onomatopoeia Confusion Matrix", "ono_conf_matrix.png")

# classification report 출력
print("Label Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=label_names))

print("Onomatopoeia Classification Report:")
print(classification_report(true_onos, pred_onos, target_names=ono_names))

print(f"Results saved to {save_dir}")
