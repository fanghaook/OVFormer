import torch
from clip import clip
from PIL import Image
from tqdm import tqdm
import json
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
for _, param in model.named_parameters():
    param.requires_grad = False

# LVIS train
json_path = 'datasets/lvis/lvis_v1_train.json'
file_dir = "datasets/coco/train2017/"
save_path = "datasets/metadata/lvis_train_clip_feature.pkl"
data = json.load(open(json_path, 'r'))
dic = {}
for image in tqdm(data['images']):
    file_name = file_dir + f"{image['id']}".zfill(12) + ".jpg"
    image_clip = preprocess(Image.open(file_name)).unsqueeze(0).to(device)
    feature_clip = model.encode_image(image_clip)
    dic[image['id']] = feature_clip
torch.save(dic, save_path)


# LVIS val
json_path = 'datasets/lvis/lvis_v1_val.json'
file_dir = "datasets/coco/val2017/"
save_path = "datasets/metadata/lvis_val_clip_feature.pkl"
data = json.load(open(json_path, 'r'))
dic = {}
for image in tqdm(data['images']):
    file_name = file_dir + f"{image['id']}".zfill(12) + ".jpg"
    if not os.path.exists(file_name):
        file_name = "datasets/coco/train2017/" + f"{image['id']}".zfill(12) + ".jpg"
    image_clip = preprocess(Image.open(file_name)).unsqueeze(0).to(device)
    feature_clip = model.encode_image(image_clip)
    dic[image['id']] = feature_clip
torch.save(dic, save_path)

# LVVIS train
json_path = 'datasets/LVVIS/train/train_instances_.json'   #
file_dir = "datasets/LVVIS/train/JPEGImages/"
save_path = "datasets/metadata/lvvis_train_clip_feature.pkl"
data = json.load(open(json_path, 'r'))
dic = {}
for video in tqdm(data['videos']):
    for image in video["file_names"]:
        file_name = file_dir + image
        image_clip = preprocess(Image.open(file_name)).unsqueeze(0).to(device)
        feature_clip = model.encode_image(image_clip)
        dic[file_name] = feature_clip
torch.save(dic, save_path)

# LVVIS val
json_path = 'datasets/LVVIS/val/val_instances_.json'
file_dir = "datasets/LVVIS/val/JPEGImages/"
save_path = "datasets/metadata/lvvis_val_clip_feature.pkl"
data = json.load(open(json_path, 'r'))
dic = {}
for video in tqdm(data['videos']):
    for image in video["file_names"]:
        file_name = file_dir + image
        image_clip = preprocess(Image.open(file_name)).unsqueeze(0).to(device)
        feature_clip = model.encode_image(image_clip)
        dic[file_name] = feature_clip
torch.save(dic, save_path)

# LVVIS test
json_path = 'datasets/LVVIS/test/test_instances.json'
file_dir = "datasets/LVVIS/test/JPEGImages/"
save_path = "datasets/metadata/lvvis_test_clip_feature.pkl"
data = json.load(open(json_path, 'r'))
dic = {}
for video in tqdm(data['videos']):
    for image in video["file_names"]:
        file_name = file_dir + image
        image_clip = preprocess(Image.open(file_name)).unsqueeze(0).to(device)
        feature_clip = model.encode_image(image_clip)
        dic[file_name] = feature_clip
torch.save(dic, save_path)

# ytvis_2019 val
json_path = 'datasets/ytvis_2019/valid.json'
file_dir = "datasets/ytvis_2019/valid/JPEGImages/"
save_path = "datasets/metadata/ytvis_2019_val_clip_feature.pkl"
data = json.load(open(json_path, 'r'))
dic = {}
for video in tqdm(data['videos']):
    for image in video["file_names"]:
        file_name = file_dir + image
        image_clip = preprocess(Image.open(file_name)).unsqueeze(0).to(device)
        feature_clip = model.encode_image(image_clip)
        dic[file_name] = feature_clip
torch.save(dic, save_path)

# ytvis_2021 val
json_path = 'datasets/ytvis_2021/valid.json'
file_dir = "datasets/ytvis_2021/valid/JPEGImages/"
save_path = "datasets/metadata/ytvis_2021_val_clip_feature.pkl"
data = json.load(open(json_path, 'r'))
dic = {}
for video in tqdm(data['videos']):
    for image in video["file_names"]:
        file_name = file_dir + image
        image_clip = preprocess(Image.open(file_name)).unsqueeze(0).to(device)
        feature_clip = model.encode_image(image_clip)
        dic[file_name] = feature_clip
torch.save(dic, save_path)

# ovis val
json_path = 'datasets/ovis/annotations/valid.json'
file_dir = "datasets/ovis/valid/"
save_path = "datasets/metadata/ovis_val_clip_feature.pkl"
data = json.load(open(json_path, 'r'))
dic = {}
for video in tqdm(data['videos']):
    for image in video["file_names"]:
        file_name = file_dir + image
        image_clip = preprocess(Image.open(file_name)).unsqueeze(0).to(device)
        feature_clip = model.encode_image(image_clip)
        dic[file_name] = feature_clip
torch.save(dic, save_path)

# burst val
json_path = 'datasets/burst/b2y_val.json'
file_dir = "datasets/burst/val/"
save_path = "datasets/metadata/burst_val_clip_feature.pkl"
data = json.load(open(json_path, 'r'))
dic = {}
for video in tqdm(data['videos']):
    for image in video["file_names"]:
        file_name = file_dir + image
        image_clip = preprocess(Image.open(file_name)).unsqueeze(0).to(device)
        feature_clip = model.encode_image(image_clip)
        dic[file_name] = feature_clip
torch.save(dic, save_path)
