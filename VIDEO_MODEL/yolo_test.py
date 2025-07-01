import os
import shutil
import glob
import json
import yaml

from tqdm import tqdm
import random

from PIL import Image

import cv2
import matplotlib.pyplot as plt

# YOLO
from ultralytics import YOLO

# custom
from data_collector import DataCollector

# Data Paths, 속도 저하 이슈가 있어서 잠시 주석처리. 나중에 전체 데이터에 대한 테스트 수행을 원할시 해제.

# test_root_path = ''
# save_path = ''

# # Collect each iamge and label path as list

# image_data = glob.glob(f"{test_root_path}\\image\\**\\*.jpg", recursive=True)
# image_annot_data = glob.glob(f"{test_root_path}\\label\\**\\*.json", recursive=True)

# # It returns right [img_path, annot_path] pair
# test_data = DataCollector(image_data, image_annot_data)

# Image Copy and Move
# Convert img name from Ko to En
# Annotation to YOLO

# Labels are set automatically, so this code can be reused when adding more data

def labelExtracter(json_path):
    with open(json_path, encoding='utf-8') as f:
        meta = json.load(f)
    return meta['info']['class']['depth2']

def labelMapper(json_paths):
    labels = sorted(list(set([labelExtracter(p) for p in json_paths])))
    return {label: idx for idx, label in enumerate(labels)}

# Copy image and make YOLO format annotations

def cimyfa(save_path, label_map):
    for phase, data in zip(['test'], [test_data]):
        print(f"{phase} data processing...")

        img_dir_path = os.path.join(save_path, 'images', phase)
        annot_dir_path = os.path.join(save_path, 'labels', phase)

        os.makedirs(img_dir_path, exist_ok=True)
        os.makedirs(annot_dir_path, exist_ok=True)

        for idx, pair in enumerate(tqdm(data)):

            img_path = pair[0]
            annot_path = pair[1]

            img_name = f"{phase}_{idx}.jpg"
            annot_name = f"{phase}_{idx}.txt"

            dest_path = os.path.join(img_dir_path, img_name)

            shutil.copy(img_path, dest_path)

            with open(annot_path, encoding='utf-8') as f:
                meta = json.load(f)

            image = Image.open(dest_path)
            w, h = image.size

            yolo_labels = []

            for ann in meta["annotations"]:
                bbox = ann.get("bbox", None)
                if not bbox:
                    continue

                x = bbox["bndex_xcrdnt"]
                y = bbox["bndex_ycrdnt"]
                bw = bbox["bndex_width"]
                bh = bbox["bndex_hg"]

                # 중심 좌표 및 정규화
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h

                class_name = meta['info']['class']['depth2']
                class_id = label_map[class_name]

                yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            # 저장 경로 결정
            label_path = os.path.join(annot_dir_path, annot_name)

            with open(label_path, "w", encoding="utf-8") as out_file:
                out_file.write("\n".join(yolo_labels))
        
        print(f"Finished processing {phase} data!")


# AI hub의 데이터를 YOLO 형식으로 변환해 저장하는 코드. 이미 실행 했을경우 사용 X
def data_prepare(image_data, image_annot_data, save_path):
    print(f"test len* img:{len(image_data)}, label:{len(image_annot_data)}")
    
    if len(image_data) != len(image_annot_data):
        print("The number of image data and the number of annotation data is not equal.")
        raise

    label_map = labelMapper(image_annot_data)
    print(label_map)
    cimyfa(save_path=save_path, label_map=label_map)

if __name__ == '__main__':

    model = YOLO('./runs/detect/train4/weights/best.pt')
    
    img_path = ''

    for folder in os.listdir(img_path):
        folder_path = os.path.join(img_path, folder)
        
        if os.path.isdir(folder_path):
            
            image_files = [f for f in os.listdir(folder_path) if os.path.basename(f).split('.')[1]=="jpg"]

            if image_files:
                selected_file = random.choice(image_files)
                selected_path = os.path.join(folder_path, selected_file)

                results = model(selected_path)

                image = results[0].plot()
                iamge = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
                plt.imshow(iamge)
                plt.title(selected_file)
                plt.axis('off')
                plt.show()

    #results = model.val(data='_yolo.yaml', split='test')

    # print(results.box.map, results.box.map50)
    # print(results.box.maps)
