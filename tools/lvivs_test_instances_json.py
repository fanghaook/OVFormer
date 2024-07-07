import os
import json
import cv2

val_instances = json.load(open('datasets/LVVIS/val/val_instances_.json', 'r'))
categories = val_instances['categories']

videos = []
video_folder = 'datasets/lvvis/test/JPEGImages'
video_ids = sorted(os.listdir(video_folder))
for video_id, video_name in enumerate(video_ids):
    video_path = os.path.join(video_folder, video_name)
    file_names = sorted(os.listdir(video_path))

    first_frame_path = os.path.join(video_path, file_names[0])
    frame = cv2.imread(first_frame_path)
    height, width, _ = frame.shape

    video_info = {
        'id': video_id,
        'width': width,
        'height': height,
        'length': len(file_names),
        'file_names': [os.path.join(video_name, file_name) for file_name in file_names]
    }
    videos.append(video_info)

test_data = {
    'videos': videos,
    'categories': categories
}

with open('datasets/LVVIS/test/test_instances.json', 'w') as f:
    json.dump(test_data, f, indent=4)

print("test_instances.json done.")