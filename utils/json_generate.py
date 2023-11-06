import os
import json

data_path = "/root/autodl-tmp/BraTs2018/BraTs2018"


data = {
    "training": [],
}
import random
length = 285
elements = [0, 1, 2, 3, 4]

# 计算每个元素应该出现的次数
element_count = length // len(elements)
result = []
for element in elements:
    result.extend([element] * element_count)

# 打乱列表
random.shuffle(result)

fold_mapping = {
    "MICCAI_BraTS_2018_Data_Training/HGG": 0,
    "MICCAI_BraTS_2018_Data_Training/LGG": 0,
    # "MICCAI_BraTS_2018_Data_Validation": 1
}

for fold_dir, fold_name in fold_mapping.items():
    fold_path = os.path.join(data_path, fold_dir)
    if os.path.isdir(fold_path):
        for patient in os.listdir(fold_path):
            patient_path = os.path.join(fold_path, patient)
            patient_path_save = patient_path.replace(data_path+'/',"")
            if patient_path.endswith('.csv'):
                break
            image_files = [f for f in os.listdir(patient_path) if f.endswith(".nii")]
            image_paths = [os.path.join(patient_path_save, f) for f in image_files  if not f.endswith("_seg.nii")]
            label_path = [os.path.join(patient_path_save, f) for f in image_files  if f.endswith("_seg.nii")]
            sample = result.pop()
            data_entry = {
                "fold": sample,
                "image": image_paths,
                "label": label_path
            }
            data['training'].append(data_entry)

output_json = "/root/SwimUNETR_miss/BraTS2018/jsons/BraTS2018_data.json"
with open(output_json, "w") as json_file:
    json.dump(data, json_file, indent=4)
