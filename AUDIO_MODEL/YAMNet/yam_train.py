import os
import glob
import json

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import MelSpectrogram

from tqdm import tqdm

from torch_vggish_yamnet import yamnet
from data_collector import DataCollector

root_path = ''

sound_data = glob.glob(f"{root_path}\\train\\audio\\**\\*.wav", recursive=True)
sound_annot_data = glob.glob(f"{root_path}\\train\\label\\**\\*.json", recursive=True)
val_sound_data = glob.glob(f"{root_path}\\valid\\audio\\**\\*.wav", recursive=True)
val_sound_annot_data = glob.glob(f"{root_path}\\valid\\label\\**\\*.json", recursive=True)

train_data = DataCollector(sound_data, sound_annot_data)
val_data = DataCollector(val_sound_data, val_sound_annot_data)

print(f"Train data size: {len(train_data)}, Valid data size: {len(val_data)}")

class CustomSoundDataset(Dataset):
    def __init__(self, json_wav_pairs, win_length=512, sample_rate=16000, n_mels=128, n_fft=1024, max_len=1000):
        self.data = json_wav_pairs  # 리스트 형태: [[json_path, [wav_path]], ...]
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.max_len = max_len  # 최대 길이 제한

        # MelSpectrogram 변환
        self.mel_transform = MelSpectrogram(sample_rate=self.sample_rate, n_mels=self.n_mels, n_fft=self.n_fft, win_length=self.win_length)

        # 라벨 매핑 (depth2 기준)
        self.labels = sorted(list(set([self._get_label(json_path) for json_path, _ in self.data])))
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

    def _get_label(self, json_path):
        with open(json_path, encoding='utf-8') as f:
            meta = json.load(f)
        return meta['info']['class']['depth2']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        json_path, wav_paths = self.data[idx]
        wav_path = wav_paths[0]

        # 라벨 추출
        label_name = self._get_label(json_path)
        label_idx = self.label_to_idx[label_name]
        
        # 오디오 불러오기
        waveform, sr = torchaudio.load(wav_path)

        # mono 변환 (stereo 대응)
        if waveform.shape[0] > 1:  # 2D라면
            waveform = waveform.mean(dim=0, keepdim=True)

        # MelSpectrogram 변환
        mel_spec = self.mel_transform(waveform)  # [1, n_mels, time]

        # 패딩: 최대 길이보다 짧은 경우 패딩, 긴 경우 크롭
        mel_spec_len = mel_spec.shape[-1]
        if mel_spec_len < self.max_len:
            pad_size = self.max_len - mel_spec_len
            mel_spec = nn.functional.pad(mel_spec, (0, pad_size))  # 뒤쪽에 패딩 추가
        elif mel_spec_len > self.max_len:
            mel_spec = mel_spec[:, :, :self.max_len]  # 시간 축에 대해서 크롭

        return mel_spec, label_idx

if __name__=='__main__':
    # 데이터 준비
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.amp import autocast, GradScaler

    with open("yam_hyper_param.json", 'r') as f:
         buff = json.load(f)
    hyper_params = buff["params"]

    train_dataset = CustomSoundDataset(train_data)
    val_dataset = CustomSoundDataset(val_data)
    print(f"Data_class: {train_dataset.label_to_idx}")

    train_dataloader = DataLoader(train_dataset, batch_size=hyper_params["batch"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hyper_params["batch"], shuffle=False)
    # {'굽기': 0, '끓이기': 1, '냉장고 사용': 2, '드라이어 사용': 3, '믹서기 사용': 4, '썰기': 5, '압력밥솥 사용': 6, '전자레인지 사용': 7, '튀기기': 8}
    
    # 모델 준비
    model = yamnet.yamnet()
    model.classifier = nn.Linear(in_features=1024, out_features=len(train_dataset.label_to_idx), bias=True)

    try:
        if os.path.exists('yam_best_model.pth'):
            model.load_state_dict(torch.load('yam_best_model.pth'))
            print("yam_best_model.pth loaded")
    except Exception as e:
        print(e)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # 학습 구성
    criterion = nn.CrossEntropyLoss(label_smoothing=hyper_params["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=hyper_params["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=hyper_params["epoch"], eta_min=hyper_params["eta_min"])
    scaler = GradScaler(device=device_name)

    num_epoch = hyper_params["epoch"]
    accumulation_steps = hyper_params["accum"]
    best_acc = buff["best_acc"]

    # 🔸 검증 함수 정의
    def evaluate(model, dataloader):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                embeddings, logits = model(inputs)
                
                loss = criterion(logits, labels)
                total_loss += loss.item()

                _, preds = torch.max(logits, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(dataloader), 100 * correct / total

    # 학습 루프
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast(device_type=device_name):
                embeddings, logits = model(inputs)
                loss = criterion(logits, labels)
                loss = loss / accumulation_steps
            
    
            scaler.scale(loss).backward()

            if (i+1) % accumulation_steps == 0:
                scaler.step(optimizer)
                model.zero_grad()
                scaler.update()
                optimizer.zero_grad()
                running_loss += loss.item() * accumulation_steps

        train_loss = running_loss / len(train_dataloader)

        # 🔸 validation 평가
        val_loss, val_acc = evaluate(model, val_dataloader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")
        
        # 🔸 모델 저장 조건
        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'yam_best_model.pth')
            print(f"✔️ Best model saved (Val Loss: {val_loss:.4f})")
            
            buff["best_acc"] = val_acc
            with open("yam_hyper_param.json", 'w') as f:
                json.dump(buff, f)

        scheduler.step()

    # 🔹 최종 Train Accuracy 출력
    train_acc_loss, train_acc = evaluate(model, train_dataloader)
    print(f"Train Accuracy: {train_acc:.2f}%")
