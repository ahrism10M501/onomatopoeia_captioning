import json
import os
import glob
import chardet
from tqdm import tqdm

class DataCollector():
    def __init__(self, data_path, annot_data):
        self.data_path = data_path
        self.annot_data = annot_data
    
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self.data_path)))]

        data = self.data_path[idx]
        data_name = os.path.basename(data).split('.')[0]
        annot = [anno for anno in self.annot_data if data_name in anno]
        return [annot, data]

# 자동 인코딩 감지 후 JSON 읽기
def read_json_with_encoding_detection(path):
    try:
        with open(path, 'rb') as f:
            raw_data = f.read()
            encoding = chardet.detect(raw_data)['encoding']
        return json.loads(raw_data.decode(encoding))
    except Exception as e:
        print(f"[Error] {path} 읽기 실패 | 인코딩: {encoding} | 에러: {e}")
        return None

# 경로 설정
root_path = ""
labels = glob.glob(f"{root_path}/train/label/**/*.json", recursive=True) + glob.glob(f"{root_path}/valid/label/**/*.json", recursive=True) + glob.glob(f"{root_path}/test/label/**/*.json", recursive=True) 
print(f"총 JSON 라벨 수: {len(labels)}")

# 의성어 목록
ono_map = {
    0: "Sizzle", 1: "Woooong", 2: "ai_speaking", 3: "tack-",
    4: "glug glug", 5: "Whirrrrl", 6: "chzzzzzz", 7: "Whuuuuu", 8: "drrrrrr"
}

# 클러스터 반복
for idx in range(0, 9):
    ono = ono_map.get(idx)
    print(f"\n▶ 클러스터 {idx}: {ono} 라벨링 중...")

    clustered_data = glob.glob(f"{root_path}/clustering/cluster_{idx}/*.wav")
    data_pair = DataCollector(clustered_data, labels)

    for annot_list, audio_path in tqdm(data_pair):
        if not annot_list:
            print(f"해당 오디오에 대한 라벨 없음: {audio_path}")
            continue

        label_path = annot_list[0]  # 첫 번째 매칭만 사용

        buff = read_json_with_encoding_detection(label_path)
        if buff is None or "info" not in buff:
            print(f"건너뜀 (형식 오류 또는 info 없음): {label_path}")
            continue

        # 의성어 라벨 추가
        buff["info"]["onomatopoeia"] = ono

        # 덮어쓰기 (UTF-8)
        with open(label_path, "w", encoding="utf-8") as j:
            json.dump(buff, j, indent=0, ensure_ascii=False)
