import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm


obj_dict_path = "/home/myw/wuchangli/yk/my_ov/video_attn/github_version/dataset/VSPW_480p/label_num_dic_final.json"
with open(obj_dict_path, "r") as f:
    obj_dict = json.load(f)

# id to name
id_to_name = {}
for key, value in obj_dict.items():
    id_to_name[value] = key

vspw_root = "/home/myw/wuchangli/yk/my_ov/video_attn/github_version/dataset/VSPW_480p"
data_path = os.path.join(vspw_root, "data")
val_txt = os.path.join(vspw_root, "val.txt")

val_video_list = []
with open(val_txt, "r") as f:
    for line in f.readlines():
        video_name = line.strip().split()[0]
        val_video_list.append(video_name)
print(len(val_video_list))


video_semantic_dict = {}
for video_name in tqdm(val_video_list):
    video_semantic_set = set()
    video_path = os.path.join(data_path, video_name)
    if not os.path.exists(video_path):
        print(f"Video path {video_path} does not exist.")
        continue
    mask_path = os.path.join(video_path, "mask")
    # origin_path = os.path.join(video_path, "origin")
    mask_pngs = os.listdir(mask_path)
    mask_pngs.sort()
    for mask_img in mask_pngs:
        mask_img_path = os.path.join(mask_path, mask_img)
        # mask_img_path = "/home/myw/wuchangli/yk/my_ov/video_attn/github_version/dataset/VSPW_480p/data/0_wHveSGjXyDY/mask/00000166.png"
        mask_img_pil = Image.open(mask_img_path)
        mask_img_np = np.array(mask_img_pil)
        unique_values = np.unique(mask_img_np)
        for value in unique_values:
            if value > 124:
                # import pdb; pdb.set_trace()
                continue
            unique_cls = id_to_name[str(value)]
            video_semantic_set.add(unique_cls)
    video_semantic_dict[video_name] = list(video_semantic_set)
    # import pdb; pdb.set_trace()
save_path = os.path.join(vspw_root, "val_video_semantic.json")
with open(save_path, "w") as f:
    json.dump(video_semantic_dict, f, indent=4)
print(f"Video semantic dictionary saved to {save_path}")




     




