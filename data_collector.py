import os
import glob

train_root_path = ''

image_data = glob.glob(f"{train_root_path}\\image\\**\\*.jpg", recursive=True)
image_annot_data = glob.glob(f"{train_root_path}\\label\\**\\*.json", recursive=True)

def printData():
    return [len(image_annot_data), len(image_data)]
    
class DataCollector():
    def __init__(self, data_path, annot_data):
        self.data_path = data_path
        self.annot_data = annot_data
    
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self.annot_data)))]
        
        annot = self.annot_data[idx]
        annot_name = os.path.basename(annot).split('.')[0]

        data = [image for image in self.data_path if annot_name in image]

        output = []
        output.extend(data)
        output.append(annot)

        return output # [data_path, annot_path]
    
if __name__=='__main__':
    data_path = DataCollector(image_data, image_annot_data)
    print(data_path[2])
    print(printData())