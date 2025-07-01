import os
import numpy as np
import time
import queue
import threading
import sounddevice as sd

import torch
import torchaudio
import torchaudio.functional as F

from ultralytics import YOLO
import cv2

# 프로젝트 모듈
from models.Audio_labeler import SoundClassifier # (오디오 분류, 의성어) 모델
import sub_animation as anim # 사용자 정의 텍스트 애니메이션 모듈

# 파라미터 정의
NUM_CLASS = 9 # 인식할 비디오, 오디오 물체 수
NUM_ONO = 9 # 의성어 수

VIDEO_WEIGHT = "models/video_m.pt"
AUDIO_WEIGHT = "models/audio.pth"
VERBOSE = False # YOLOv8 모델 verbose 모드
RETRY_NUM = 5 # 모델 가중치 로드 재시도 횟수

WIDTH, HEIGHT = 800, 600
FRAME_RATE = 30

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 5.0 # 음성 인식 임계값, dB 단위, 4.0 ~ 9.0 사이로 조정. 잡음 제거.
SILENCE_TIME = 0.1 # 초 단위, threshold 이상의 소리가 이 시간보다 지속될떄 음성으로 간주

# 애니메이션 정의
seq_anim = anim.Sequential([
            anim.Parallel([anim.Fade(anim="in", duration=0.3, alpha=1.0),
                           anim.Movement(anim="up", duration=0.3, vel=50, angle=30)]),
            anim.Parallel([anim.Movement(anim="up", duration=1.5, vel=50, angle=20),
                           anim.Vibration(anim="side", duration=1.0, hz=2, amp=5),
                           anim.Fade(anim="out", duration=1.0, alpha=1.0, smooth=True)])
        ]) 

# 클래스 및 의성어 매핑
cls_mapping = {
    0:'굽기', 1:'끓이기', 2:'냉장고 사용', 3:'드라이어 사용', 4:'믹서기 사용',
    5:'썰기', 6:'압력밥솥 사용', 7:'전자레인지 사용', 8:'튀기기'
}
onos_mapping = {
    0: "Sizzle", 1: "Woooong", 2: "ai_speaking", 3: "tack-",
    4: "glug glug", 5: "Whirrrrl", 6: "chzzzzzz", 7: "Whuuuuu", 8: "drrrrrr"
}

# 모델 불러오기
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
sound_model = SoundClassifier(num_classes=NUM_CLASS, num_oto=NUM_ONO)

for i in range(1, RETRY_NUM+1):
    try:
        if os.path.exists(AUDIO_WEIGHT):
            sound_model.load_state_dict(torch.load(AUDIO_WEIGHT, map_location=device))
            print("audio model weights loaded")
            break
        else:
            print(f"Try {i}: Failed to load audio weigths")
            print(f"There isn't exist Weights file. Check the 'audio_weight_path'")
            
    except Exception as e:
        print(f"Try {i}: Failed to load audio weigths")
        print(f"Error: {e}")

sound_model.to(device)

for i in range(1, RETRY_NUM+1):
    try:
        video_model = YOLO(VIDEO_WEIGHT, verbose=VERBOSE) # YOLOv8 모델 불러오기
    except:
        print(f"Try {i}: Failed to load video weights")
        print(f'Error: {e}')

# 애니메이션 텍스트 정의
class AnimationText():
    def __init__(self, text, animation: anim, box: anim.point_box):
        self.text = text
        self.anim = animation.clone()
        self.start_pos = box.random_point()
        
    def step(self):
        result = self.anim.step()
        dx, dy = result["pos_offset"]
        alpha = result["alpha"]
        return (int(self.start_pos[0] + dx), int(self.start_pos[1] + dy)), alpha
    
    def is_finished(self):
        return self.anim.is_finished()
    
def render_text(frame, text, position, alpha, font_scale=1, thickness=1, color=(0, 0, 0)):
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = int(position[0] - text_width / 2)
    text_y = int(position[1] + text_height / 2)
    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def putAnimationText(frame, text, pt1, pt2, anim):
    bbox = (pt1[0], pt1[1], pt2[0], pt2[1])
    anim_text = AnimationText(text, bbox, anim)
    pos, alpha = anim_text.step()
    if pos is not None:
        return render_text(frame, text, pos, alpha)
    return frame

# 실시간 소리 예측 
sound_model.eval()

audio_q = queue.Queue()
last_audio_result = (None, None)

def audio_callback(indata, frames, time_info, status):
    audio_q.put(indata.copy())

def audio_thread():
    global last_audio_result
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=SAMPLE_RATE):
        while True:
            if not audio_q.empty():
                audio_block = audio_q.get()
                mono = np.mean(audio_block, axis=1)
                waveform = torch.tensor(mono, dtype=torch.float32).unsqueeze(0)

                if not F.vad(waveform, SAMPLE_RATE, trigger_time=SILENCE_TIME, trigger_level=SILENCE_THRESHOLD).any():
                    last_audio_result = (None, None)
                    print(last_audio_result)
                    continue

                transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64)
                features = transform(waveform)
                features = torchaudio.transforms.AmplitudeToDB()(features)
                features = features.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    label_out, ono_out = sound_model(features)
                    _, label_idx = torch.max(label_out, 1)
                    _, ono_idx = torch.max(ono_out, 1)
                    last_audio_result = (cls_mapping[label_idx.item()], onos_mapping[ono_idx.item()])
                    print(last_audio_result)
            else:
                time.sleep(0.01)

# 실시간 비디오 예측 및 의성어 텍스트 삽입
def video_loop():
    texts = []
    delay_ms = int(1000/FRAME_RATE)
    cap = cv2.VideoCapture(0)

    global last_audio_result

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = video_model(frame, verbose=VERBOSE)[0]
        boxes = results[0].boxes if len(results.boxes) else None

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = results[0].names[int(box.cls[0])]
                confidence = box.conf[0].item()

                if confidence > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    width = int(abs(x1 - x2))
                    height = int(abs(y1 - y2))

                    if last_audio_result[0] is not None:
                        class_name_audio, onomatopoeia = last_audio_result    
                        bbox = anim.point_box(cx, cy, width, height)
                        anim_text = AnimationText(str(onomatopoeia), seq_anim, bbox)
                        texts.append(anim_text)

        for t in texts:
            if not t.is_finished():
                pos, alpha = t.step()
                frame = render_text(frame, t.text, pos, alpha)

        texts = [t for t in texts if not t.is_finished()]

        cv2.imshow("SOAMATCH", frame)
        key = cv2.waitKey(delay_ms) & 0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
# 메인 실행
if __name__ == '__main__':
    threading.Thread(target=audio_thread, daemon=True).start()
    video_loop()