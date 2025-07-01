import os
import glob
import json

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import MelSpectrogram

from tqdm import tqdm

import Audio_labeler
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
        self.labels = sorted(list(set([self._get_label(json_path)[0] for json_path, _ in self.data])))
        self.onos = sorted(list(set([self._get_label(json_path)[1] for json_path, _ in self.data])))

        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.ono_to_idx = {ono: i for i, ono in enumerate(self.onos)}
        
    def _get_label(self, json_path):
        with open(json_path, encoding='utf-8') as f:
            meta = json.load(f)
        return [meta['info']['class']['depth2'], meta['info']['onomatopoeia']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        json_path, wav_paths = self.data[idx]
        wav_path = wav_paths[0]

        # 라벨 추출
        label_name, ono = self._get_label(json_path)
        label_idx = self.label_to_idx[label_name]
        ono_idx = self.ono_to_idx[ono]

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

        return mel_spec, label_idx, ono_idx

if __name__=='__main__':
    # 데이터 준비
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.amp import autocast, GradScaler

    with open("hyper_param2.json", 'r') as f:
         buff = json.load(f)
    hyper_params = buff["params"]

    train_dataset = CustomSoundDataset(train_data)
    val_dataset = CustomSoundDataset(val_data)
    print(f"Data_class: {train_dataset.label_to_idx}")
    print(f"Ono_class: {train_dataset.ono_to_idx}")

    train_dataloader = DataLoader(train_dataset, batch_size=hyper_params["batch"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hyper_params["batch"], shuffle=False)
    # {'굽기': 0, '끓이기': 1, '냉장고 사용': 2, '드라이어 사용': 3, '믹서기 사용': 4, '썰기': 5, '압력밥솥 사용': 6, '전자레인지 사용': 7, '튀기기': 8}
    
    # 모델 준비
    model = Audio_labeler.SoundClassifier(num_classes=len(train_dataset.label_to_idx), num_oto=len(train_dataset.ono_to_idx))

    try:
        if os.path.exists('best_model2.pth'):
            model.load_state_dict(torch.load('best_model2.pth'))
            print("best_model2.pth loaded")
    except Exception as e:
        print(e)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # 학습 구성
    criterion = nn.CrossEntropyLoss(label_smoothing=hyper_params["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=hyper_params["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=hyper_params["epoch"], eta_min=hyper_params["eta_min"])
    scaler = GradScaler()

    num_epoch = hyper_params["epoch"]
    accumulation_steps = hyper_params["accum"]
    best_label_acc = buff["best_label_acc"]
    best_ono_acc = buff["best_ono_acc"]

    # 🔸 검증 함수 정의
    def evaluate(model, dataloader):
        model.eval()
        total_loss = 0.0
        label_correct = 0
        label_total = 0
        oto_correct = 0
        oto_total = 0

        with torch.no_grad():
            for inputs, labels, otos in dataloader:
                inputs, labels, otos = inputs.to(device), labels.to(device), otos.to(device)

                outputs, oto_out = model(inputs)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(oto_out, otos)
                loss = loss1+loss2
                total_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                label_correct += (preds == labels).sum().item()
                label_total += labels.size(0)

                _, oto_preds = torch.max(oto_out, 1)
                oto_correct += (oto_preds == otos).sum().item()
                oto_total += otos.size(0)

        return total_loss / len(dataloader), 100 * label_correct / label_total, 100*  oto_correct / oto_total

    # 학습 루프
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels, otos) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            inputs, labels, otos = inputs.to(device), labels.to(device), otos.to(device)

            with autocast(device_type=device_name):
                label_pred, oto_pred = model(inputs)
                label_loss = criterion(label_pred, labels)
                oto_loss = criterion(oto_pred, otos)
                loss = (label_loss+oto_loss) / accumulation_steps
            
            scaler.scale(loss).backward()

            if (i+1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                running_loss += loss.item() * accumulation_steps

        train_loss = running_loss / len(train_dataloader)

        # 🔸 validation 평가
        val_loss, label_acc, ono_acc = evaluate(model, val_dataloader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, label Acc = {label_acc:.2f}%, ono acc = {ono_acc:.2f}")
        
        # 🔸 모델 저장 조건
        if best_label_acc < label_acc and best_ono_acc < ono_acc:
            best_label_acc = label_acc
            best_ono_acc = ono_acc
            torch.save(model.state_dict(), 'best_model2.pth')
            print(f"✔️ Best model saved (Val Loss: {val_loss:.4f})")
            buff["best_label_acc"] = label_acc
            buff["best_ono_acc"] = ono_acc
            with open("hyper_param2.json", 'w') as f:
                json.dump(buff, f)

        scheduler.step()

    # 🔹 최종 Train Accuracy 출력
    train_acc_loss, train_acc, ono_tacc = evaluate(model, train_dataloader)
    print(f"Train Accuracy: {train_acc:.2f}%, {ono_tacc}%")
