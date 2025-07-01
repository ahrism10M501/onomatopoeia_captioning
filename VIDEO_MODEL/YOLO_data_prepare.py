import os
import shutil
import glob
import json
import yaml
from PIL import Image
from tqdm import tqdm

# YOLO
from ultralytics import YOLO

# custom
from data_collector import DataCollector

# Data Paths

root_path = ''
save_path = f'{root_path}\\yolo_data'

# Collect each iamge and label path as list

image_data = glob.glob(f"{root_path}\\train\\image\\**\\*.jpg", recursive=True)
image_annot_data = glob.glob(f"{root_path}\\train\\label\\**\\*.json", recursive=True)
val_image_data = glob.glob(f"{root_path}\\valid\\image\\**\\*.jpg", recursive=True)
val_image_annot_data = glob.glob(f"{root_path}\\valid\\label\\**\\*.json", recursive=True)

# It returns right [img_path, annot_path] pair
train_data = DataCollector(image_data, image_annot_data)
val_data = DataCollector(val_image_data, val_image_annot_data)

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
    for phase, data in zip(['train', 'val'], [train_data, val_data]):
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

if __name__ == '__main__':

    print(f"train len* img:{len(image_data)}, label:{len(image_annot_data)}")
    print(f"valid len* img:{len(val_image_data)}, label:{len(val_image_annot_data)}")
    
    if len(image_data) != len(image_annot_data) or len(val_image_data) != len(val_image_annot_data):
        print("The number of image data and the number of annotation data is not equal.")
        raise

    label_map = labelMapper(image_annot_data + val_image_annot_data)
    print(label_map)
    cimyfa(save_path=save_path, label_map=label_map)

    yaml_dict = {
        'train': os.path.join(save_path, 'images/train'),
        'val': os.path.join(save_path, 'images/val'),
        'nc': len(label_map),
        'names': [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    }

    with open(os.path.join(root_path, '_yolo.yaml'), 'w') as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)
